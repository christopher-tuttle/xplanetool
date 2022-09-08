use clap::{arg, command, Command};
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let matches = command!()
        .propagate_version(true)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(arg!(-v --verbose ... "Increases verbosity by one level (can be repeated)"))
        .arg(
            arg!(--vmodule <FILTER> "sets the env_logger filter to the given string: \
                                     default,module::path=level,... \
                                     {trace,debug,info,warn,error,none}")
            .required(false),
        )
        .subcommand(Command::new("version").about("Print version information"))
        .subcommand(
            Command::new("dsf")
                .about("prints details about a dsf file")
                .arg(arg!(-a - -all "If set, prints out verbose details."))
                .arg(arg!([PATH]).required(true).allow_invalid_utf8(true)),
        )
        .subcommand(
            Command::new("dsf2text")
                .about("prints out the given dsf file as text")
                .arg(arg!([INPUT]).required(true).allow_invalid_utf8(true))
                .arg(arg!([OUTPUT]).required(false).allow_invalid_utf8(true)),
        )
        .subcommand(
            Command::new("png")
                .about("extracts the rasters as pngs")
                .arg(arg!([INPUT]).required(true).allow_invalid_utf8(true)),
        )
        .get_matches();

    // Logging stuff: keep above everything but the cmdline parsing.
    let mut log_builder = env_logger::Builder::new();
    match matches.occurrences_of("verbose") {
        1 => {
            log_builder.filter_level(log::LevelFilter::Info);
        }
        2 => {
            log_builder.filter_level(log::LevelFilter::Debug);
        }
        3 => {
            log_builder.filter_level(log::LevelFilter::Trace);
        }
        _ => (),
    };
    if let Some(filter) = matches.value_of("vmodule") {
        log_builder.parse_filters(&filter);
    }

    match matches.subcommand() {
        Some(("version", _)) => {
            println!(env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(("dsf", sub_matches)) => {
            let path = sub_matches.value_of_os("PATH").unwrap();
            let info = dsf::DsfInfo::new(&path)?;
            if sub_matches.is_present("all") {
                // verbose
                println!("{:+}", &info); // Use {:+} for verbose printing.
            } else {
                println!("{}", &info);
            }
            // TODO: Make the DSF api more rich to print out the strings.
        }
        Some(("dsf2text", sub_matches)) => {
            let in_path = sub_matches.value_of_os("INPUT").unwrap();
            let info = dsf::DsfInfo::new(&in_path)?;
            let text = info.to_text()?;
            if let Some(out_path) = sub_matches.value_of_os("OUTPUT") {
                let mut out = std::fs::File::create(out_path)?;
                use std::io::Write;
                out.write_all(&text.as_bytes())?;
                println!("Wrote data to {:?}", &out_path);
            } else {
                println!("{}", text);
            }
        }
        Some(("png", sub_matches)) => {
            let in_path = sub_matches.value_of_os("INPUT").unwrap();
            xplanetool::exp_pngs(&in_path)?;
        }
        _ => unreachable!()
    }
    Ok(())
}
