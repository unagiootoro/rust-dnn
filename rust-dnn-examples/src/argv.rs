use std::{collections::HashMap, env};

pub fn get_argv() -> HashMap<String, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut map = HashMap::new();
    for arg in args {
        if let Some((key, value)) = split_once(&arg, '=') {
            map.insert(key.to_string(), value.to_string());
        } else {
            panic!("引数 '{}' は key=value 形式ではありません", arg);
        }
    }
    map
}

fn split_once(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut iter = s.splitn(2, delimiter);
    let key = iter.next()?;
    let value = iter.next()?;
    Some((key, value))
}
