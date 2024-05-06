// Implementation of Fast Walsh Transforms (FWTs) using sequency and
// Hadamard in-place transforms.  Both algorithms are O(n log(n)).
//
// Note that these transforms are their own inverse, to within a scale
// factor of vector.length, because the transform matrix is orthogonal and
// symmetric about its diagonal.

use std::ops::Add;
use std::ops::Sub;

// Perform a fast Walsh transformation using Manz sequency ordering.
pub fn sequency<T>(input_v: &[T]) -> Option<Vec<T>>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>,
{
    let length = input_v.len();
    let mut v = input_v.to_vec();

    if power_of_2(length.try_into().unwrap()) {
        let mut j = 0;
        for i in 0..(length - 2) {
            if i < j {
                (v[i], v[j]) = (v[j], v[i]);
            }
            let mut k = length >> 1;
            while k <= j {
                j -= k;
                k >>= 1;
            }
            j += k;
        }
        let mut offset = length;
        while offset > 1 {
            let lag = offset >> 1;
            let ngroups = length / offset;
            for group in 0..ngroups {
                for i in 0..lag {
                    j = i + group * offset;
                    let k = j + lag;
                    if group & 1 == 1 {
                        (v[j], v[k]) = (v[j] - v[k], v[j] + v[k]);
                    } else {
                        (v[j], v[k]) = (v[j] + v[k], v[j] - v[k]);
                    }
                }
            }
            offset = lag;
        }
        Some(v)
    } else {
        None
    }
}

// Perform a fast Walsh transformation using Hadamard (natural) ordering.
pub fn hadamard<T>(input_v: &[T]) -> Option<Vec<T>>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>,
{
    let length = input_v.len();
    let mut v = input_v.to_vec();
    if power_of_2(length.try_into().unwrap()) {
        let mut lag = 1;
        while lag < length {
            let offset = lag << 1;
            let ngroups = length / offset;
            for group in 0..ngroups {
                for base in 0..lag {
                    let j = base + group * offset;
                    let k = j + lag;
                    (v[j], v[k]) = (v[j] + v[k], v[j] - v[k]);
                }
            }
            lag = offset;
        }
        Some(v)
    } else {
        None
    }
}

// Boolean test of whether an Integer is a pure power of two.
// This is an O(1) algorithm.
pub fn power_of_2(n: usize) -> bool {
    match n {
        0 => false,
        _ => n & (n - 1) == 0,
    }
}

// Scale a vector by its length
pub fn scale<T>(v: &[T]) -> Vec<f64>
where
    T: Copy,
    f64: From<T>,
{
    let length: usize = v.len();
    let result: Vec<f64> = v
        .iter()
        .map(|x| <T as TryInto<f64>>::try_into(*x).unwrap() as f64 / (length as f64))
        .collect();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let input_v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let input_v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let input_v = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]);
        let input_v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]);
        let input_v = [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
        ];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(
            result,
            [
                1., -1., -1., 1., -1., 1., 1., -1., -1., 1., 1., -1., 1., -1., -1., 1.
            ]
        );
        let input_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let result = hadamard(&input_v).unwrap();
        assert_eq!(
            result,
            [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]
        );
    }

    #[test]
    fn test_sequency() {
        let input_v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let input_v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let input_v = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]);
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(result, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let input_v = [
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ];
        let result = sequency(&input_v).unwrap();
        assert_eq!(
            result,
            [1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1., -1., -1.]
        );
        let input_v = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = sequency(&input_v).unwrap();
        assert_eq!(
            result,
            [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
        );
    }

    #[test]
    fn test_scaling() {
        let input = vec![3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&input), outcome);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let outcome = vec![0.25, 0.5, 0.75, 1.0];
        assert_eq!(scale(&input), outcome);
        let input = [3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&input), outcome);
    }

    #[test]
    fn test_power_of_2() {
        assert!(power_of_2(1));
        assert!(power_of_2(2));
        assert!(power_of_2(4));
        assert!(power_of_2(8));
        assert!(power_of_2(16));
        assert!(power_of_2(1 << 31));
        assert!(!power_of_2(0));
        assert!(!power_of_2(3));
        assert!(!power_of_2(5));
        assert!(!power_of_2(7));
        assert!(!power_of_2(usize::MAX));
    }

    #[test]
    fn test_non_power_of_2_seq() {
        assert_eq!(sequency(&[0.0, 0.0, 1.0]), None);
    }

    #[test]
    fn test_non_power_of_2_had() {
        assert_eq!(hadamard(&[0.0, 0.0, 1.0]), None);
    }

    #[test]
    fn test_empty_array() {
        let v : Vec<i32> = [].to_vec();
        assert_eq!(sequency(&v), None);
    }
}
