mod embed;
mod input;
mod output;
mod reorder;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "ignite-ms",
    version,
    about = "High-throughput GPU text embedding engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed text from input files
    Embed(embed::EmbedArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Embed(args) => embed::run(args)?,
    }
    Ok(())
}
