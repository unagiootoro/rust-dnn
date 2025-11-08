use std::cell::RefCell;

thread_local! {
    pub static RANDOM: RefCell<XorShift128Plus> = RefCell::new(XorShift128Plus::from_seed(42));
}

// xorshift128plus ランダム生成器
// 暗号用途には使用しないこと

/// splitmix64: シード拡張用
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// xorshift128+ 生成器構造体
#[derive(Clone, Debug)]
pub struct XorShift128Plus {
    s0: u64,
    s1: u64,
}

impl XorShift128Plus {
    /// シードから初期化。両方が 0 になることは避ける。
    pub fn from_seed(seed: u64) -> Self {
        let s0 = splitmix64(seed);
        let s1 = splitmix64(s0);
        let (s0, s1) = if s0 == 0 && s1 == 0 {
            (0x9e3779b97f4a7c15u64, 0x6a09e667f3bcc909u64)
        } else {
            (s0, s1)
        };
        XorShift128Plus { s0, s1 }
    }

    pub fn from_seed2() -> Self {
        let seed = RANDOM.with(|random| random.borrow_mut().next_u64());
        Self::from_seed(seed)
    }

    /// 次の乱数（u64）
    pub fn next_u64(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        let result = s0.wrapping_add(s1);

        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0;
        s1 ^= s0 >> 26;

        self.s0 = s0;
        self.s1 = s1;
        result
    }

    /// 次の乱数（0.0〜1.0未満の f64）
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeatability() {
        // 同じシードなら同じ結果を返す
        let mut rng1 = XorShift128Plus::from_seed(0xDEADBEEFCAFEBABE);
        let mut rng2 = XorShift128Plus::from_seed(0xDEADBEEFCAFEBABE);

        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seed() {
        // 異なるシードではほぼ確実に違う結果
        let mut rng1 = XorShift128Plus::from_seed(1);
        let mut rng2 = XorShift128Plus::from_seed(2);

        let seq1: Vec<u64> = (0..5).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..5).map(|_| rng2.next_u64()).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_not_zero_state() {
        // シードが 0 の場合でも内部状態が (0, 0) にならない
        let rng = XorShift128Plus::from_seed(0);
        assert!(!(rng.s0 == 0 && rng.s1 == 0));
    }

    #[test]
    fn test_range_of_f64() {
        // next_f64() が 0.0 <= x < 1.0 の範囲にある
        let mut rng = XorShift128Plus::from_seed(1234);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!(x >= 0.0 && x < 1.0, "x = {}", x);
        }
    }

    #[test]
    fn test_sequence_changes() {
        // 値が変化していく（同じ値が連続しないことを確認）
        let mut rng = XorShift128Plus::from_seed(42);
        let first = rng.next_u64();
        let second = rng.next_u64();
        assert_ne!(first, second);
    }
}
