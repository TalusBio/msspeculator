use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=MSSPECULATOR_GIT_COMMIT");
    let commit = std::env::var("MSSPECULATOR_GIT_COMMIT").ok().or_else(|| {
        Command::new("git")
            .args(["rev-parse", "--verify", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_string())
    });
    println!(
        "cargo:rustc-env=MSSPECULATOR_GIT_COMMIT={}",
        commit.as_deref().unwrap_or("unknown")
    );
}
