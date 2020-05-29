use crate::matrix::Matrix;
use crate::mds::SparseMatrix;
use crate::poseidon::{Arity, PoseidonConstants};

use bellperson::gadgets::boolean::Boolean;
use bellperson::gadgets::num;
use bellperson::gadgets::num::AllocatedNum;
use bellperson::{ConstraintSystem, LinearCombination, SynthesisError};
use ff::Field;
use ff::ScalarEngine as Engine;
use std::marker::PhantomData;

/// Similar to `num::Num`, we use `Elt` to accumulate both values and linear combinations, then eventually
/// extract into a `num::AllocatedNum`, enforcing that the linear combination corresponds to the result.
/// In this way, all intermediate calculations are accounted for, with the restriction that we can only
/// accumulate linear (not polynomial) constraints. The set of operations provided here ensure this invariant is maintained.
#[derive(Clone)]
enum Elt<E: Engine> {
    Allocated(AllocatedNum<E>),
    Num(num::Num<E>),
}

impl<E: Engine> Elt<E> {
    fn is_allocated(&self) -> bool {
        if let Self::Allocated(_) = self {
            true
        } else {
            false
        }
    }

    fn is_num(&self) -> bool {
        if let Self::Num(_) = self {
            true
        } else {
            false
        }
    }

    fn num_from_fr<CS: ConstraintSystem<E>>(fr: E::Fr) -> Self {
        let num = num::Num::<E>::zero();
        Self::Num(num.add_bool_with_coeff(CS::one(), &Boolean::Constant(true), fr))
    }

    fn ensure_allocated<CS: ConstraintSystem<E>>(
        &self,
        cs: &mut CS,
        enforce: bool,
    ) -> Result<AllocatedNum<E>, SynthesisError> {
        match self {
            Self::Allocated(v) => Ok(v.clone()),
            Self::Num(num) => {
                let v = AllocatedNum::alloc(cs.namespace(|| "allocate for Elt::Num"), || {
                    num.get_value()
                        .ok_or_else(|| SynthesisError::AssignmentMissing)
                })?;

                if enforce {
                    cs.enforce(
                        || format!("enforce num allocation preserves lc"),
                        |_| num.lc(E::Fr::one()),
                        |lc| lc + CS::one(),
                        |lc| lc + v.get_variable(),
                    );
                }
                Ok(v)
            }
        }
    }

    fn val(&self) -> Option<E::Fr> {
        match self {
            Self::Allocated(v) => v.get_value(),
            Self::Num(num) => num.get_value(),
        }
    }

    fn lc(&self) -> LinearCombination<E> {
        match self {
            Self::Num(num) => num.lc(E::Fr::one()),
            Self::Allocated(v) => LinearCombination::<E>::zero() + v.get_variable(),
        }
    }

    /// Add two Nums and return a Num tracking the calculation. It is forbidden to invoke on an Allocated because the intended computation
    /// does not include that path.
    fn add<CS: ConstraintSystem<E>>(self, other: Elt<E>) -> Result<Elt<E>, SynthesisError> {
        match (self, other) {
            (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(&b))),
            _ => panic!("only two numbers may be added"),
        }
    }

    /// Scale
    fn scale<CS: ConstraintSystem<E>>(self, scalar: E::Fr) -> Result<Elt<E>, SynthesisError> {
        match self {
            Elt::Num(num) => Ok(Elt::Num(num.scale(scalar))),
            Elt::Allocated(a) => Elt::Num(a.into()).scale::<CS>(scalar),
        }
    }
}

/// Circuit for Poseidon hash.
pub struct PoseidonCircuit<'a, E, A>
where
    E: Engine,
    A: Arity<E::Fr>,
{
    constants_offset: usize,
    width: usize,
    elements: Vec<Elt<E>>,
    pos: usize,
    current_round: usize,
    constants: &'a PoseidonConstants<E, A>,
    _w: PhantomData<A>,
}

/// PoseidonCircuit implementation.
impl<'a, E, A> PoseidonCircuit<'a, E, A>
where
    E: Engine,
    A: Arity<E::Fr>,
{
    /// Create a new Poseidon hasher for `preimage`.
    pub fn new(
        allocated_nums: Vec<AllocatedNum<E>>,
        constants: &'a PoseidonConstants<E, A>,
    ) -> Self {
        let width = constants.width();
        let elements = allocated_nums.into_iter().map(Elt::Allocated).collect();
        PoseidonCircuit {
            constants_offset: 0,
            width,
            elements,
            pos: width,
            current_round: 0,
            constants,
            _w: PhantomData::<A>,
        }
    }

    fn hash<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
    ) -> Result<AllocatedNum<E>, SynthesisError> {
        self.full_round(cs.namespace(|| "first round"), true, false)?;

        for i in 1..self.constants.full_rounds / 2 {
            self.full_round(
                cs.namespace(|| format!("initial full round {}", i)),
                false,
                false,
            )?;
        }

        for i in 0..self.constants.partial_rounds {
            self.partial_round(cs.namespace(|| format!("partial round {}", i)))?;
        }

        for i in 0..(self.constants.full_rounds / 2) - 1 {
            self.full_round(
                cs.namespace(|| format!("final full round {}", i)),
                false,
                false,
            )?;
        }
        self.full_round(cs.namespace(|| "terminal full round"), false, true)?;

        self.elements[1].ensure_allocated(&mut cs.namespace(|| "hash result"), true)
    }

    fn full_round<CS: ConstraintSystem<E>>(
        &mut self,
        mut cs: CS,
        first_round: bool,
        last_round: bool,
    ) -> Result<(), SynthesisError> {
        let mut constants_offset = self.constants_offset;

        let pre_round_keys = if first_round {
            (0..self.width)
                .map(|i| self.constants.compressed_round_constants[constants_offset + i])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        constants_offset += pre_round_keys.len();

        let post_round_keys = if first_round || !last_round {
            (0..self.width)
                .map(|i| self.constants.compressed_round_constants[constants_offset + i])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        constants_offset += post_round_keys.len();

        // Apply the quintic S-Box to all elements
        for i in 0..self.elements.len() {
            let pre_round_key = if first_round {
                let rk = pre_round_keys[i];
                Some(rk)
            } else {
                None
            };

            let post_round_key = if first_round || !last_round {
                let rk = post_round_keys[i];
                Some(rk)
            } else {
                None
            };

            if first_round {
                self.elements[i] = quintic_s_box_pre_add(
                    cs.namespace(|| format!("quintic s-box {}", i)),
                    &self.elements[i],
                    pre_round_key,
                    post_round_key,
                )?;
            } else {
                self.elements[i] = quintic_s_box(
                    cs.namespace(|| format!("quintic s-box {}", i)),
                    &self.elements[i],
                    post_round_key,
                )?;
            }
        }
        self.constants_offset = constants_offset;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>()?;
        Ok(())
    }

    fn partial_round<CS: ConstraintSystem<E>>(&mut self, mut cs: CS) -> Result<(), SynthesisError> {
        let round_key = self.constants.compressed_round_constants[self.constants_offset];
        self.constants_offset += 1;
        // Apply the quintic S-Box to the first element.
        self.elements[0] = quintic_s_box(
            cs.namespace(|| "solitary quintic s-box"),
            &self.elements[0],
            Some(round_key),
        )?;

        // Multiply the elements by the constant MDS matrix
        self.product_mds::<CS>()?;
        Ok(())
    }

    fn product_mds_m<CS: ConstraintSystem<E>>(&mut self) -> Result<(), SynthesisError> {
        self.product_mds_with_matrix::<CS>(&self.constants.mds_matrices.m)
    }

    /// Set the provided elements with the result of the product between the elements and the appropriate
    /// MDS matrix.
    fn product_mds<CS: ConstraintSystem<E>>(&mut self) -> Result<(), SynthesisError> {
        let full_half = self.constants.half_full_rounds;
        let sparse_offset = full_half - 1;
        if self.current_round == sparse_offset {
            self.product_mds_with_matrix::<CS>(&self.constants.pre_sparse_matrix)?;
        } else {
            if (self.current_round > sparse_offset)
                && (self.current_round < full_half + self.constants.partial_rounds)
            {
                let index = self.current_round - sparse_offset - 1;
                let sparse_matrix = &self.constants.sparse_matrixes[index];

                self.product_mds_with_sparse_matrix::<CS>(&sparse_matrix)?;
            } else {
                self.product_mds_m::<CS>()?;
            }
        };

        self.current_round += 1;
        Ok(())
    }

    fn product_mds_with_matrix<CS: ConstraintSystem<E>>(
        &mut self,
        matrix: &Matrix<E::Fr>,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<Elt<E>> = Vec::with_capacity(self.constants.width());

        for j in 0..self.constants.width() {
            let column = (0..self.constants.width())
                .map(|i| matrix[i][j])
                .collect::<Vec<_>>();

            let product = scalar_product::<E, CS>(self.elements.as_slice(), &column)?;

            result.push(product);
        }

        self.elements = result;

        Ok(())
    }

    // Sparse matrix in this context means one of the form, M''.
    fn product_mds_with_sparse_matrix<CS: ConstraintSystem<E>>(
        &mut self,
        matrix: &SparseMatrix<E>,
    ) -> Result<(), SynthesisError> {
        let mut result: Vec<Elt<E>> = Vec::with_capacity(self.constants.width());

        result.push(scalar_product::<E, CS>(
            self.elements.as_slice(),
            &matrix.w_hat,
        )?);

        for j in 1..self.width {
            result.push(
                self.elements[j].clone().add::<CS>(
                    self.elements[0]
                        .clone() // First row is dense.
                        .scale::<CS>(matrix.v_rest[j - 1])?, // Except for first row/column, diagonals are one.
                )?,
            );
        }

        self.elements = result;

        Ok(())
    }

    fn debug(&self) {
        let element_frs: Vec<_> = self.elements.iter().map(|n| n.val()).collect::<Vec<_>>();
        dbg!(element_frs, self.constants_offset);
    }
}

/// Create circuit for Poseidon hash.
pub fn poseidon_hash<CS, E, A>(
    mut cs: CS,
    mut preimage: Vec<AllocatedNum<E>>,
    constants: &PoseidonConstants<E, A>,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
    E: Engine,
    A: Arity<E::Fr>,
{
    // Add the arity tag to the front of the preimage.
    let tag = constants.arity_tag;
    let tag_num = AllocatedNum::alloc(cs.namespace(|| "arity tag"), || Ok(tag))?;
    preimage.push(tag_num);
    preimage.rotate_right(1);

    let mut p = PoseidonCircuit::new(preimage, constants);

    p.hash(cs)
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to result.
fn quintic_s_box<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    e: &Elt<E>,
    post_round_key: Option<E::Fr>,
) -> Result<Elt<E>, SynthesisError> {
    let l = e.ensure_allocated(&mut cs.namespace(|| "S-box input"), true)?;

    // If round_key was supplied, add it after all exponentiation.
    let l2 = l.square(cs.namespace(|| "l^2"))?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(
        cs.namespace(|| "(l4 * l) + rk)"),
        &l4,
        &l,
        None,
        post_round_key,
        true,
    );

    Ok(Elt::Allocated(l5?))
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box_pre_add<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    e: &Elt<E>,
    pre_round_key: Option<E::Fr>,
    post_round_key: Option<E::Fr>,
) -> Result<Elt<E>, SynthesisError> {
    if let (Some(pre_round_key), Some(post_round_key)) = (pre_round_key, post_round_key) {
        let l = e.ensure_allocated(&mut cs.namespace(|| "S-box input"), true)?;

        // If round_key was supplied, add it to l before squaring.
        let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), pre_round_key, &l, true)?;
        let l4 = l2.square(cs.namespace(|| "l^4"))?;
        let l5 = mul_sum(
            cs.namespace(|| "l4 * (l + rk)"),
            &l4,
            &l,
            Some(pre_round_key),
            Some(post_round_key),
            true,
        );

        Ok(Elt::Allocated(l5?))
    } else {
        panic!("pre_round_key and post_round_key must both be provided.");
    }
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    to_add: E::Fr,
    num: &AllocatedNum<E>,
    enforce: bool,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
{
    let res = AllocatedNum::alloc(cs.namespace(|| "squared sum"), || {
        let mut tmp = num
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp.square();

        Ok(tmp)
    })?;

    if enforce {
        cs.enforce(
            || "squared sum constraint",
            |lc| lc + num.get_variable() + (to_add, CS::one()),
            |lc| lc + num.get_variable() + (to_add, CS::one()),
            |lc| lc + res.get_variable(),
        );
    }
    Ok(res)
}

/// Calculates (a * (pre_add + b)) + post_add — and enforces that constraint.
pub fn mul_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &AllocatedNum<E>,
    pre_add: Option<E::Fr>,
    post_add: Option<E::Fr>,
    enforce: bool,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
{
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        if let Some(x) = pre_add {
            tmp.add_assign(&x);
        }
        tmp.mul_assign(
            &a.get_value()
                .ok_or_else(|| SynthesisError::AssignmentMissing)?,
        );
        if let Some(x) = post_add {
            tmp.add_assign(&x);
        }

        Ok(tmp)
    })?;

    if enforce {
        if let Some(x) = post_add {
            let mut neg = E::Fr::zero();
            neg.sub_assign(&x);

            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul sum constraint pre-post-add",
                    |lc| lc + b.get_variable() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint post-add",
                    |lc| lc + b.get_variable(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable() + (neg, CS::one()),
                );
            }
        } else {
            if let Some(pre) = pre_add {
                cs.enforce(
                    || "mul sum constraint pre-add",
                    |lc| lc + b.get_variable() + (pre, CS::one()),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable(),
                );
            } else {
                cs.enforce(
                    || "mul sum constraint",
                    |lc| lc + b.get_variable(),
                    |lc| lc + a.get_variable(),
                    |lc| lc + res.get_variable(),
                );
            }
        }
    }
    Ok(res)
}

/// Calculates a * (b + to_add) — and enforces that constraint.
pub fn mul_pre_sum<CS: ConstraintSystem<E>, E: Engine>(
    mut cs: CS,
    a: &AllocatedNum<E>,
    b: &AllocatedNum<E>,
    to_add: E::Fr,
    enforce: bool,
) -> Result<AllocatedNum<E>, SynthesisError>
where
    CS: ConstraintSystem<E>,
{
    let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
        let mut tmp = b
            .get_value()
            .ok_or_else(|| SynthesisError::AssignmentMissing)?;
        tmp.add_assign(&to_add);
        tmp.mul_assign(
            &a.get_value()
                .ok_or_else(|| SynthesisError::AssignmentMissing)?,
        );

        Ok(tmp)
    })?;

    if enforce {
        cs.enforce(
            || "mul sum constraint",
            |lc| lc + b.get_variable() + (to_add, CS::one()),
            |lc| lc + a.get_variable(),
            |lc| lc + res.get_variable(),
        );
    }
    Ok(res)
}

fn scalar_product_with_add<E: Engine, CS: ConstraintSystem<E>>(
    elts: &[Elt<E>],
    scalars: &[E::Fr],
    to_add: E::Fr,
) -> Result<Elt<E>, SynthesisError> {
    let tmp = scalar_product::<E, CS>(elts, scalars)?;
    let tmp2 = tmp.add::<CS>(Elt::<E>::num_from_fr::<CS>(to_add))?;

    Ok(tmp2)
}

fn scalar_product<E: Engine, CS: ConstraintSystem<E>>(
    elts: &[Elt<E>],
    scalars: &[E::Fr],
) -> Result<Elt<E>, SynthesisError> {
    elts.iter()
        .zip(scalars)
        .try_fold(Elt::Num(num::Num::zero()), |acc, (elt, &scalar)| {
            acc.add::<CS>(elt.clone().scale::<CS>(scalar)?)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon::HashMode;
    use crate::test::TestConstraintSystem;
    use crate::{scalar_from_u64, Poseidon, Strength};
    use bellperson::ConstraintSystem;
    use generic_array::typenum;
    use paired::bls12_381::{Bls12, Fr};
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_poseidon_hash() {
        test_poseidon_hash_aux::<typenum::U2>(Strength::Standard, 314);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Standard, 380);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Standard, 508);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Standard, 764);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Standard, 1012);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Standard, 1388);

        test_poseidon_hash_aux::<typenum::U2>(Strength::Strengthened, 370);
        test_poseidon_hash_aux::<typenum::U4>(Strength::Strengthened, 436);
        test_poseidon_hash_aux::<typenum::U8>(Strength::Strengthened, 568);
        test_poseidon_hash_aux::<typenum::U16>(Strength::Strengthened, 824);
        test_poseidon_hash_aux::<typenum::U24>(Strength::Strengthened, 1072);
        test_poseidon_hash_aux::<typenum::U36>(Strength::Strengthened, 1448);
    }

    fn test_poseidon_hash_aux<A>(strength: Strength, expected_constraints: usize)
    where
        A: Arity<<Bls12 as Engine>::Fr>,
    {
        let mut rng = XorShiftRng::from_seed(crate::TEST_SEED);
        let mut cs = TestConstraintSystem::<Bls12>::new();
        let arity = A::to_usize();
        let constants = PoseidonConstants::<Bls12, A>::new_with_strength(strength);

        let expected_constraints_calculated = {
            let width = 1 + arity;
            let s_boxes = (width * constants.full_rounds) + constants.partial_rounds;
            let s_box_constraints = 3 * s_boxes;
            let mds_constraints =
                (width * constants.full_rounds) + constants.partial_rounds - arity;
            let total_constraints = s_box_constraints + mds_constraints;

            total_constraints
        };
        let mut i = 0;

        let mut fr_data = vec![Fr::zero(); arity];
        let data: Vec<AllocatedNum<Bls12>> = (0..arity)
            .enumerate()
            .map(|_| {
                let fr = Fr::random(&mut rng);
                fr_data[i] = fr;
                i += 1;
                AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(fr)).unwrap()
            })
            .collect::<Vec<_>>();

        let out = poseidon_hash(&mut cs, data, &constants).expect("poseidon hashing failed");

        let mut p = Poseidon::<Bls12, A>::new_with_preimage(&fr_data, &constants);
        let expected: Fr = p.hash_in_mode(HashMode::Correct);

        assert!(cs.is_satisfied(), "constraints not satisfied");

        assert_eq!(
            expected,
            out.get_value().unwrap(),
            "circuit and non-circuit do not match"
        );

        assert_eq!(
            expected_constraints_calculated,
            cs.num_constraints(),
            "constraint number miscalculated"
        );

        assert_eq!(
            expected_constraints,
            cs.num_constraints(),
            "constraint number changed",
        );
    }

    fn fr(n: u64) -> <Bls12 as Engine>::Fr {
        scalar_from_u64::<<Bls12 as Engine>::Fr>(n)
    }

    fn efr(n: u64) -> Elt<Bls12> {
        Elt::num_from_fr::<TestConstraintSystem<Bls12>>(fr(n))
    }

    #[test]
    fn test_square_sum() {
        let mut cs = TestConstraintSystem::<Bls12>::new();

        let mut cs1 = cs.namespace(|| "square_sum");
        let two = fr(2);
        let three =
            AllocatedNum::alloc(cs1.namespace(|| "three"), || Ok(scalar_from_u64(3))).unwrap();
        let res = square_sum(cs1, two, &three, true).unwrap();

        let twenty_five: Fr = scalar_from_u64(25);
        assert_eq!(twenty_five, res.get_value().unwrap());
    }

    #[test]
    fn test_scalar_product() {
        {
            // Inputs are all linear combinations.
            let two = efr(2);
            let three = efr(3);
            let four = efr(4);

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            assert!(res.is_num());
            assert_eq!(scalar_from_u64::<Fr>(56), res.val().unwrap());
        }
        {
            let mut cs = TestConstraintSystem::<Bls12>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 =
                AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64(3))).unwrap();
            let three = Elt::Allocated(n3.clone());
            let n4 =
                AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64(4))).unwrap();
            let four = Elt::Allocated(n4.clone());

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            assert!(res.is_num());
            assert_eq!(scalar_from_u64::<Fr>(56), res.val().unwrap());

            res.lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(6));
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(7));
                };
            });

            res.ensure_allocated(&mut cs, true).unwrap();
            assert!(cs.is_satisfied());
        }
        {
            let mut cs = TestConstraintSystem::<Bls12>::new();

            // Inputs are linear combinations and an allocated number.
            let two = efr(2);

            let n3 =
                AllocatedNum::alloc(cs.namespace(|| "three"), || Ok(scalar_from_u64(3))).unwrap();
            let three = Elt::Allocated(n3.clone());
            let n4 =
                AllocatedNum::alloc(cs.namespace(|| "four"), || Ok(scalar_from_u64(4))).unwrap();
            let four = Elt::Allocated(n4.clone());

            let mut res_vec = Vec::new();

            let res = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &[two, three, four],
                &[fr(5), fr(6), fr(7)],
            )
            .unwrap();

            res_vec.push(res);

            assert!(res_vec[0].is_num());
            assert_eq!(fr(56), res_vec[0].val().unwrap());

            res_vec[0].lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(6)); // 6 * three
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(7)); // 7 * four
                };
            });

            let four2 = Elt::Allocated(n4.clone());
            res_vec.push(efr(3));
            res_vec.push(four2);
            let res2 = scalar_product::<Bls12, TestConstraintSystem<Bls12>>(
                &res_vec,
                &[fr(7), fr(8), fr(9)],
            )
            .unwrap();

            res2.lc().iter().for_each(|(var, f)| {
                if var.get_unchecked() == n3.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(42)); // 7 * 6 * three
                };
                if var.get_unchecked() == n4.get_variable().get_unchecked() {
                    assert_eq!(*f, fr(58)); // (7 * 7 * four) + (9 * four)
                };
            });

            let allocated = res2.ensure_allocated(&mut cs, true).unwrap();

            let v = allocated.get_value().unwrap();
            assert_eq!(fr(452), v); // (7 * 56) + (8 * 3) + (9 * 4) = 448

            assert!(cs.is_satisfied());
        }
    }

    #[test]
    fn test_scalar_product_with_add() {
        let two = efr(2);
        let three = efr(3);
        let four = efr(4);

        let res = scalar_product_with_add::<Bls12, TestConstraintSystem<Bls12>>(
            &[two, three, four],
            &[fr(5), fr(6), fr(7)],
            fr(3),
        )
        .unwrap();

        assert!(res.is_num());
        assert_eq!(fr(59), res.val().unwrap());
    }
}
