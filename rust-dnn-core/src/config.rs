use std::cell::RefCell;

thread_local! {
    static CONFIG: RefCell<Config> = RefCell::new(Config::new());
}

struct Config {
    enable_backprop: bool,
}

impl Config {
    fn new() -> Self {
        Self {
            enable_backprop: true,
        }
    }
}

pub fn enable_backprop() -> bool {
    CONFIG.with(|config| config.borrow().enable_backprop)
}

pub fn set_enable_backprop(enable: bool) {
    CONFIG.with(|config| config.borrow_mut().enable_backprop = enable);
}
