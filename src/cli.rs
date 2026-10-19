//! Command-line options of the binaries: every option has the form `--name value`.

use std::fmt::Debug;
use std::str::FromStr;

/// The command-line arguments of a binary.
pub struct Args(Vec<String>);

impl Args {
    pub fn from_env() -> Self {
        Args(std::env::args().collect())
    }

    /// The value after `--name`, if the option is given.
    pub fn get(&self, name: &str) -> Option<String> {
        self.0.iter().position(|arg| arg == name).and_then(|i| self.0.get(i + 1)).cloned()
    }

    /// The value after `--name`, or `default`.
    pub fn string_or(&self, name: &str, default: &str) -> String {
        self.get(name).unwrap_or_else(|| default.to_string())
    }

    /// The value after `--name`, parsed, or `default` if the option is not given.
    pub fn parse_or<T: FromStr>(&self, name: &str, default: T) -> T
    where
        T::Err: Debug,
    {
        match self.get(name) {
            Some(value) => value.parse().unwrap_or_else(|e| panic!("{name} {value}: {e:?}")),
            None => default,
        }
    }

    /// The comma-separated values after `--name`; empty if the option is not given.
    pub fn list(&self, name: &str) -> Vec<String> {
        self.get(name).map(|value| value.split(',').map(String::from).collect()).unwrap_or_default()
    }
}

/// Is the instance selected by `filters`? A filter is an instance id ("B16_256_3") or a family ("B16_256"). An empty
/// list selects every instance.
pub fn is_selected(id: &str, filters: &[String]) -> bool {
    filters.is_empty()
        || filters.iter().any(|f| id == f || id.strip_prefix(f.as_str()).is_some_and(|rest| rest.starts_with('_')))
}
