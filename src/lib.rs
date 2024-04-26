// Implementation of Fast Walsh Transforms (FWTs) using sequency and
// Hadamard in-place transforms.  Both algorithms are O(n log(n)).
//
// Note that these transforms are their own inverse, to within a scale
// factor of vector.length, because the transform matrix is orthogonal and
// symmetric about its diagonal.

use std::ops::Add;
use std::ops::Sub;

// Perform a fast Walsh transformation using a Manz sequency ordered
// in-place algorithm. The vector is modified by this algorithm.
pub fn sequency<T>(v : &mut [T]) -> &mut [T]
where
     T : Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>
{
    // let v: &[f64] = v.as_ref();
    let length = v.len();
    if !power_of_2(length.try_into().unwrap()) {
        panic!("sequency: vector length must be a power of 2")
    }
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
    v
}

// Perform a fast Walsh transformation using a Hadamard (natural) ordered
// in-place algorithm. The vector is modified by this algorithm.
pub fn hadamard<T>(v : &mut [T]) -> &mut [T]
where
     T : Add<Output = T> + Sub<Output = T> + Copy + std::ops::AddAssign<T>
{
    // let v: &[f64] = v.as_ref();
    let length = v.len();
    if !power_of_2(length.try_into().unwrap()) {
        panic!("hadamard: vector length must be a power of 2")
    }
    let mut lag = 1;
    let length = v.len();
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
    v
}

// Boolean test of whether an Integer is a pure power of two.
// This is an O(1) algorithm.
pub fn power_of_2(n: u32) -> bool {
    match n {
        0 => false,
        _ => n & (n - 1) == 0,
    }
}

// Scale a vector by its length
pub fn scale<T>(v: &mut [T]) -> Vec<f64>
where
    T: Copy,
    f64: From<T>
{
    let length : usize = v.len(); //.try_into().unwrap();
    let result : Vec<f64> =
        v.iter().map(|x|
            <T as TryInto<f64>>::try_into(*x).unwrap() as f64 / (length as f64)
        ).collect();
    result
}


#[cfg(test)]
mod tests {
    use super::*;

    //    #[test]
    //    fn it_works() {
    //        let result = add(2, 2);
    //        assert_eq!(result, 4);
    //    }

    #[test]
    fn test_hadamard() {
        assert_eq!(
            hadamard(&mut [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
        );
        assert_eq!(
            hadamard(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
        );
        assert_eq!(
            hadamard(&mut [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1]
        );
    }

    #[test]
    fn test_sequency() {
        assert_eq!(sequency(&mut [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            sequency(&mut [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(
            sequency(&mut [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        );
        assert_eq!(
            sequency(&mut [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
        );
    }

    #[test]
    fn test_scaling() {
        let mut input = vec![3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&mut input), outcome);
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        let outcome = vec![0.25, 0.5, 0.75, 1.0];
        assert_eq!(scale(&mut input), outcome);
        let mut input = [3, 6, 9];
        let outcome = vec![1.0, 2.0, 3.0];
        assert_eq!(scale(&mut input), outcome);
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
        assert!(!power_of_2(u32::MAX));
    }

    #[test]
    #[should_panic]
    fn test_non_power_of_2_seq() {
        sequency(&mut [0.0, 0.0, 1.0]);
    }
    // non-power-of-2 array
    #[test]
    #[should_panic]
    fn test_non_power_of_2_had() {
        hadamard(&mut [0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn test_empty_array() {
        sequency::<i32>(&mut []);
    }
}
