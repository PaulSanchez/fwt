#![crate_name = "fwt"]

//! Walsh transforms are useful in a variety of applications, such as image or
//! speech processing, filtering, and efficiently creating [very large statistical
//! designs of experiments](https://core.ac.uk/download/pdf/36728443.pdf).
//! 
//! Walsh functions are a binary-valued (±1) alternative to the more widely known
//! Fourier functions. Their time index can be represented using different orderings,
//! but regardless of the ordering used they constitute a complete orthogonal basis
//! for a vector space. Fast Walsh Transforms (FWTs)&mdash;similar to the well-known
//! Fast Fourier Transform&mdash;provide computationally efficient and numerically
//! stable calculations of the transform. This package provides FWT implementations
//! for sequency and Hadamard ordering. Both algorithms have O(*n* log(*n*)) time
//! complexity, where *n* is the length of the slice to be transformed and must
//! be a power of 2.
//!
//! Walsh transformations are computed solely using addition and subtraction.
//! Consequently, the output type (float vs int) conforms to the input type.
//!
//! Note that these transforms are their own inverse to within a scale
//! factor of the input slice's length.

use std::ops::Add;
use std::ops::Sub;

/// Return the Manz sequency ordering transform of `input_v`, or
/// `None` if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// let input_v = [0, 0, 0, 0, 0, 0, 1, 0];
/// let result = fwt::sequency(&input_v);
/// assert_eq!(
///     result,
///     Some(vec![1, -1, 1, -1, -1, 1, -1, 1])
/// );
/// assert_eq!(fwt::sequency(&[0.0, 0.0, 1.0]), None);
/// ```
pub fn sequency<T>(input_v: &[T]) -> Option<Vec<T>>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>,
{
    let length = input_v.len();
    let mut v = input_v.to_vec();

    if power_of_2(length) {
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

/// Return the Hadamard (natural) ordering transform of `input_v`,
/// or `None` if the input length is not a power of 2.
///
/// # Example
///
/// ```
/// let input_v = [0, 0, 0, 0, 0, 0, 0, 1];
/// let result = fwt::hadamard(&input_v);
/// assert_eq!(
///     result,
///     Some(vec![1, -1, -1, 1, -1, 1, 1, -1])
/// );
/// assert_eq!(fwt::hadamard(&[0.0, 0.0, 1.0]), None);
/// ```
pub fn hadamard<T>(input_v: &[T]) -> Option<Vec<T>>
where
    T: Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>,
{
    let length = input_v.len();
    let mut v = input_v.to_vec();
    if power_of_2(length) {
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

/// Determine whether unsigned `n` is a pure power of two, in O(1) time.
///
/// # Example
///
/// ```
/// assert!(!fwt::power_of_2(usize::MAX));
/// assert!(fwt::power_of_2(1024));
/// ```
pub fn power_of_2(n: usize) -> bool {
    match n {
        0 => false,
        _ => n & (n - 1) == 0,
    }
}

/// Scale a vector by its length. This is an appropriate scaling
/// to yield an inversion from two calls to the same transform.
/// Note that the result is `f64` even if the input `v` contains ints.
///
/// # Example
///
/// The following demonstrates that a Walsh transform is its
/// own inverse after scaling: 
/// ```
/// let input = vec![1., 2., 3., 4.];
/// let outcome = fwt::hadamard(&input).unwrap();
/// let unscaled = fwt::hadamard(&outcome).unwrap();
/// assert_eq!(input, fwt::scale(&unscaled).unwrap());
/// ```
pub fn scale<T>(v: &[T]) -> Option<Vec<f64>>
where
    T: Copy,
    f64: From<T>,
{
    let length = v.len();
    Some(v.iter()
        .map(|&x| (f64::from(x)) / (length as f64))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let input_v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        let input_v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]));
        let input_v = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]));
        let input_v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let result = hadamard(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]));
        let input_v = [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        ];
        let result = hadamard(&input_v);
        assert_eq!(
            result,
            Some(vec![1., -1., -1., 1., -1., 1., 1., -1., -1., 1., 1., -1., 1., -1., -1., 1.])
        );
        let input_v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let result = hadamard(&input_v);
        assert_eq!(
            result,
            Some(vec![1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1])
        );
    }

    #[test]
    fn test_sequency() {
        let input_v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
        let input_v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]));
        let input_v = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]));
        let input_v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let result = sequency(&input_v);
        assert_eq!(result, Some(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]));
        let input_v = [
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];
        let result = sequency(&input_v);
        assert_eq!(
            result,
            Some(vec![1., 1., 1., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1., -1., -1.])
        );
        let input_v = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = sequency(&input_v);
        assert_eq!(
            result,
            Some(vec![1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
        );
    }

    #[test]
    fn test_scaling() {
        let input = vec![3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&input), Some(outcome));
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let outcome = vec![0.25, 0.5, 0.75, 1.0];
        assert_eq!(scale(&input), Some(outcome));
        let input = [3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&input), Some(outcome));
        let input = vec![1., 2., 3., 4.];
        let outcome = hadamard(&input).unwrap();
        let unscaled = hadamard(&outcome).unwrap();
        assert_eq!(input, scale(&unscaled).unwrap());
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
    fn test_empty_array() {
        let v: Vec<i32> = [].to_vec();
        assert_eq!(sequency(&v), None);
    }
}
