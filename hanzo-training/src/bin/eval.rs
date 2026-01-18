//! Evaluation binary for trained models

use anyhow::Result;
use clap::{Parser, Subcommand};
use hanzo_training::{
    evaluation::{AccuracyBenchmark, BenchmarkRunner, PerplexityBenchmark},
    init_logging,
};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "hanzo-eval")]
#[command(about = "Evaluate Hanzo ML models")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all benchmarks
    All {
        /// Path to trained model
        #[arg(short, long, value_name = "PATH")]
        model: PathBuf,

        /// Output directory for results
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
    },

    /// Run specific benchmark
    Benchmark {
        /// Path to trained model
        #[arg(short, long, value_name = "PATH")]
        model: PathBuf,

        /// Benchmark name
        #[arg(short, long, value_name = "NAME")]
        benchmark: String,

        /// Output directory for results
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,
    },

    /// List available benchmarks
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    let cli = Cli::parse();

    match cli.command {
        Commands::All { model, output } => run_all_benchmarks(model, output).await,
        Commands::Benchmark {
            model,
            benchmark,
            output,
        } => run_single_benchmark(model, benchmark, output).await,
        Commands::List => list_benchmarks().await,
    }
}

async fn run_all_benchmarks(model_path: PathBuf, output_dir: Option<PathBuf>) -> Result<()> {
    info!("Running all benchmarks for model: {}", model_path.display());

    let mut runner = BenchmarkRunner::default();
    let results = runner.run_all()?;

    println!("\n📊 Evaluation Results");
    println!("===================");
    println!("Overall Score: {:.4}", results.overall_score);
    println!("Timestamp: {}", results.timestamp);
    println!();

    for (name, result) in &results.benchmarks {
        println!("🔹 {}: {:.4}", name, result.score);
        for (metric, value) in &result.metrics {
            println!("   {} = {:.6}", metric, value);
        }
        println!();
    }

    // Save results if output directory is specified
    if let Some(output_path) = output_dir {
        std::fs::create_dir_all(&output_path)?;
        let results_file = output_path.join("evaluation_results.json");
        let json_content = serde_json::to_string_pretty(&results)?;
        std::fs::write(&results_file, json_content)?;
        info!("Results saved to: {}", results_file.display());
    }

    Ok(())
}

async fn run_single_benchmark(
    model_path: PathBuf,
    benchmark_name: String,
    output_dir: Option<PathBuf>,
) -> Result<()> {
    info!(
        "Running {} benchmark for model: {}",
        benchmark_name,
        model_path.display()
    );

    let benchmark: Box<dyn hanzo_training::evaluation::Benchmark> = match benchmark_name.as_str() {
        "perplexity" => Box::new(PerplexityBenchmark::new()),
        "accuracy" => Box::new(AccuracyBenchmark::new()),
        _ => {
            return Err(anyhow::anyhow!("Unknown benchmark: {}", benchmark_name));
        }
    };

    let result = benchmark.run()?;

    println!("\n📊 {} Results", benchmark_name);
    println!("===================");
    println!("Score: {:.4}", result.score);
    println!();

    for (metric, value) in &result.metrics {
        println!("{} = {:.6}", metric, value);
    }

    // Save results if output directory is specified
    if let Some(output_path) = output_dir {
        std::fs::create_dir_all(&output_path)?;
        let results_file = output_path.join(format!("{}_results.json", benchmark_name));
        let json_content = serde_json::to_string_pretty(&result)?;
        std::fs::write(&results_file, json_content)?;
        info!("Results saved to: {}", results_file.display());
    }

    Ok(())
}

async fn list_benchmarks() -> Result<()> {
    println!("Available Benchmarks:");
    println!("====================");
    println!("🔹 perplexity - Measures model perplexity on evaluation dataset");
    println!("🔹 accuracy - Measures model accuracy on evaluation tasks");
    println!();
    println!("Usage:");
    println!("  hanzo-eval all --model ./path/to/model");
    println!("  hanzo-eval benchmark --model ./path/to/model --benchmark perplexity");

    Ok(())
}
