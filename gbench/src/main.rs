use bellperson::bls::Fr;
use ff::Field;
use generic_array::sequence::GenericSequence;
use generic_array::typenum::{U11, U8};
use generic_array::GenericArray;
use log::info;
use neptune::batch_hasher::BatcherType;
use neptune::column_tree_builder::{ColumnTreeBuilder, ColumnTreeBuilderTrait};
use neptune::error::Error;
use neptune::BatchHasher;
use rust_gpu_tools::opencl::GPUSelector;
use std::result::Result;
use std::thread;
use std::time::Instant;
use structopt::StructOpt;
use neptune::poseidon::HashMode;
use rand_xorshift::XorShiftRng;
use rand_core::SeedableRng;
use bellperson::domain::Scalar;


#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "Neptune gbench", about = "Neptune benchmarking program")]
struct Opts {
    #[structopt(long = "max-tree-batch-size", default_value = "700000")]
    max_tree_batch_size: usize,

    #[structopt(long = "max-column-batch-size", default_value = "400000")]
    max_column_batch_size: usize,

    #[structopt(long = "sector-size", default_value = "2KiB")]
    sector_size: String,

    // zero_input: all columns data is 0x000....000
    #[structopt(long)]
    zero_input: bool,

    // one_input: all columns data is 0x000....001
    #[structopt(long)]
    one_input: bool,

    // one_input: all columns data is random
    #[structopt(long)]
    random_input: bool,

    // gpu_cpu_parallel: gpu & cpu calculate simultaneously, check each other
    #[structopt(long)]
    gpu_cpu_parallel: bool,

    //correct: use HashMode::{Correct, OptimizedStatic};
    #[structopt(long)]
    correct: bool,
}

#[derive(Debug, Clone, Copy)]
enum InputMode {
    ZERO,
    ONE,
    RANDOM
}

fn bench_column_building(
    log_prefix: &str,
    batcher_type: Option<BatcherType>,
    leaves: usize,
    max_column_batch_size: usize,
    max_tree_batch_size: usize,
    mode :HashMode,
    input_mode: InputMode
) -> Fr {
    info!("{}: Creating ColumnTreeBuilder", log_prefix);
    let mut builder = ColumnTreeBuilder::<U11, U8>::new(
        batcher_type,
        leaves,
        max_column_batch_size,
        max_tree_batch_size,
    )
    .unwrap();
    info!("{}: ColumnTreeBuilder created", log_prefix);

    // Simplify computing the expected root.
    let constant_element = match input_mode {
        InputMode::ZERO => Fr::zero(),
        InputMode::ONE => Fr::one(),
        InputMode::RANDOM => {
            let mut rng = XorShiftRng::from_seed([
                0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
                0xbc, 0xe5,
            ]);
            Fr::random(&mut rng)
        }
    };
    let constant_column = GenericArray::<Fr, U11>::generate(|_| constant_element);
    info!("constant_column: {:?}", constant_column);

    let max_batch_size = if let Some(batcher) = &builder.column_batcher {
        batcher.max_batch_size()
    } else {
        leaves
    };

    let effective_batch_size = usize::min(leaves, max_batch_size);
    info!(
        "{}: Using effective batch size {} to build columns",
        log_prefix, effective_batch_size
    );

    info!("{}: adding column batches", log_prefix);
    info!("{}: start commitment", log_prefix);
    let start = Instant::now();
    let mut total_columns = 0;
    while total_columns + effective_batch_size < leaves {
        print!(".");
        let columns: Vec<GenericArray<Fr, U11>> =
            (0..effective_batch_size).map(|_| constant_column).collect();

        let _ = builder.add_columns(columns.as_slice(), mode).unwrap();
        total_columns += columns.len();
    }
    println!();

    let final_columns: Vec<_> = (0..leaves - total_columns)
        .map(|_| GenericArray::<Fr, U11>::generate(|_| constant_element))
        .collect();
    println!("```");
    println!("final_columns: {:?}", final_columns);
    println!("```");
    println!("---");
    info!("{}: adding final column batch and building tree", log_prefix);
    let (base, res) = builder.add_final_columns(final_columns.as_slice(), mode).unwrap();
    println!("res: {:?}", res);
    println!("base: {:?}", base);

    info!("{}: end commitment", log_prefix);
    let elapsed = start.elapsed();
    info!("{}: commitment time: {:?}", log_prefix, elapsed);

    total_columns += final_columns.len();
    assert_eq!(total_columns, leaves);

    let computed_root = res[res.len() - 1];

    let expected_root = builder.compute_uniform_tree_root(final_columns[0], mode).unwrap();
    let expected_size = builder.tree_size();
    println!("---");
    println!("```");
    println!("expected_root: {}", expected_root);
    println!("computed_root: {}", computed_root);
    println!("```");

    assert_eq!(
        expected_size,
        res.len(),
        "{}: result tree was not expected size",
        log_prefix
    );
    assert_eq!(
        expected_root, computed_root,
        "{}: computed root was not the expected one",
        log_prefix
    );

    res[res.len() - 1]
}



fn main() -> Result<(), Error> {
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    unimplemented!("Running on macos is not recommended and may have bad consequences -- experiment at your own risk.");
    env_logger::init();

    let opts = Opts::from_args();
    let max_column_batch_size = opts.max_column_batch_size;
    let max_tree_batch_size = opts.max_tree_batch_size;
    let gpu_cpu_parallel = opts.gpu_cpu_parallel;
    let correct = opts.correct;
    let kib = match opts.sector_size.as_str() {
        "2KiB" => 2,
        "512MiB" => 1024 * 512,
        "32GiB" => 1024 * 1024 * 32,
        _ => 2
    };
    info!("KiB: {}", kib);
    let bytes = kib * 1024;
    let leaves = bytes / 32;
    info!("leaves: {}", leaves);
    info!("max column batch size: {}", max_column_batch_size);
    info!("max tree batch size: {}", max_tree_batch_size);

    let input_mode = if opts.zero_input {
        InputMode::ZERO
    } else if opts.one_input {
        InputMode::ONE
    } else {
        InputMode::RANDOM
    };
    info!("input_mode: {:?}", input_mode);

    // Comma separated list of GPU bus-ids
    let gpus = std::env::var("NEPTUNE_GBENCH_GPUS");
    let batcher_types = gpus
        .map(|v| {
            v.split(",")
                .map(|s| s.parse::<u32>().expect("Invalid Bus-Id number!"))
                .map(|bus_id| BatcherType::CustomGPU(GPUSelector::BusId(bus_id)))
                .collect::<Vec<_>>()
        })
        .unwrap_or(vec![BatcherType::GPU]);

    if gpu_cpu_parallel {
        for batcher_type in batcher_types {
            let mut threads = Vec::new();

            let mut cpu_res = neptune::Scalar::zero();
            let mut gpu_res= neptune::Scalar::zero();

            let log_prefix = format!("GPU[Selector: {:?}]", batcher_type);
            threads.push(thread::spawn(move || {
                info!("log_prefix: {}", log_prefix);
                *&mut gpu_res = bench_column_building(
                    &log_prefix,
                    Some(batcher_type.clone()),
                    leaves,
                    max_column_batch_size,
                    max_tree_batch_size,
                    if correct {
                        HashMode::Correct
                    } else {
                        HashMode::OptimizedStatic
                    },
                    input_mode
                );

            }));
            threads.push(thread::spawn(move || {
                let log_prefix = format!("CPU");
                info!("{}", log_prefix);
                *&mut cpu_res = bench_column_building(
                    &log_prefix,
                    None,
                    leaves,
                    max_column_batch_size,
                    max_tree_batch_size,
                    if correct {
                        HashMode::Correct
                    } else {
                        HashMode::OptimizedStatic
                    },
                    input_mode
                );

            }));
            assert_eq!(cpu_res, gpu_res, "GPU CPU execution results are inconsistent");
            for thread in threads {
                thread.join().unwrap();
            }
        }
    } else {
        let log_prefix = format!("CPU");
        info!("{}", log_prefix);
        bench_column_building(
            &log_prefix,
            None,
            leaves,
            max_column_batch_size,
            max_tree_batch_size,
            if correct {
                HashMode::Correct
            } else {
                HashMode::OptimizedStatic
            },
            input_mode
        );
    }

    info!("end");
    // Leave time to verify GPU memory usage goes to zero before exiting.
    std::thread::sleep(std::time::Duration::from_millis(15000));
    Ok(())
}
