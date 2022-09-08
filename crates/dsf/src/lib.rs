// This is a library for reading X-Plane's DSF files.
//
// For now, the primary function is to print basic metadata about the file to figure out
// what kind of information it contains. And maybe if it has dependencies or conflicts with
// other scenery.
//
// DSF - Distribution Scenery Format
//   https://developer.x-plane.com/article/dsf-file-format-specification/
//   https://developer.x-plane.com/article/dsf-usage-in-x-plane/
//   7z compressed: https://github.com/OSSystems/compress-tools-rs prob supports 7z
//     (otool -L shows version for dylibs)
//   ... might be a good idea to have DSF analysis/printing in order to better understand
//       how these meshes work.
//
// Meshes: https://developer.x-plane.com/article/understanding-and-building-dsf-base-meshes/
// Meshtool: https://developer.x-plane.com/manuals/meshtool/ ???
//
//
// LICENSE INFORMATION
//
// This was written with considerable reference to Laminar Research's xptools package, particularly
// the paths under src/DSFTools. The code is here: https://github.com/X-Plane/xptools and it is
// published under the MIT/X11 license:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 

use log::{debug, info, trace, warn};
use simple_error::bail;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fmt::Write;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Parses and yields a stream of fixed-width, little-endian values (u16 or u32).
struct RawDecoder<'a, 'dr: 'a, T> {
    reader: &'a mut DataReader<'dr>,
    // This is needed to add the T to the struct definition.
    _marker: std::marker::PhantomData<T>,
}

impl<'a, 'dr, T> RawDecoder<'a, 'dr, T> {
    /// Constructs a decoder (iterator) over the given DataReader, which is updated.
    fn new(reader: &'a mut DataReader<'dr>) -> Self {
        Self {
            reader: reader,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Readable<T> + Copy> Iterator for RawDecoder<'_, '_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.reader.done() {
            None
        } else {
            Some(self.reader.read())
        }
    }
}

/// Parses and yields a stream of run-length-encoded values.
///
/// The input reader is expected to contain zero or more runs. Each run has a one-byte
/// header, where the lower 7 bits indicate the length of the run, and the top bit
/// indicates whether the run is of the same number (0x80 set) or not:
///
/// Examples with T=u16:
///   - [3, 1, 0, 2, 0, 3, 0] => three mixed u16 values, [1, 2, 3].
///   - [133, 42, 0] => the u16 number 42 repeated 5 times (133 = 5 | 0x80).
///
struct RunLengthDecoder<'a, 'dr: 'a, T> {
    reader: &'a mut DataReader<'dr>,
    /// When a run is in progress, indicates if the run is of the same number.
    /// (If run_remaining > 0 and this is None, then it's a mixed run.)
    same_value: Option<T>,
    /// If non-zero, the iterator has a run in progress with this many values left.
    run_remaining: usize,
}

impl<'a, 'dr, T> RunLengthDecoder<'a, 'dr, T> {
    /// Constructs a decoder (iterator) over the given DataReader, which is updated.
    fn new(reader: &'a mut DataReader<'dr>) -> Self {
        Self {
            reader: reader,
            same_value: None,
            run_remaining: 0,
        }
    }
}

// Copy is required to store in the Option.
impl<T: Readable<T> + Copy> Iterator for RunLengthDecoder<'_, '_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.run_remaining > 0 {
            if let Some(v) = self.same_value {
                // A repeated run of the same value.
                self.run_remaining -= 1;
                return Some(v);
            } else {
                if self.reader.done() {
                    panic!("Corrupted input: Expected more bytes but reader is empty.");
                }
                // A run of mixed values.
                self.run_remaining -= 1;
                return Some(self.reader.read());
            }
        } else {
            if self.reader.done() {
                return None;
            }
            // Just begun or just finished a run. A one-byte header describes the next
            // run's type and length.
            let header = self.reader.read_u8();
            if self.reader.done() {
                panic!("Corrupted input: Expected more bytes but reader is empty.");
            }

            let is_same = (header & 0x80) != 0;
            let count = header & 0x7f;
            if is_same {
                self.same_value = Some(self.reader.read());
            } else {
                self.same_value = None;
            }
            self.run_remaining = count as usize;
            return self.next();
        }
    }
}

#[cfg(test)]
mod decoder_tests {
    use super::*;

    #[test]
    fn raw_u16() {
        let input: [u8; 8] = [0, 0, 1, 0, 1, 2, 255, 255];
        let mut reader = DataReader::new(&input);
        let decoder = RawDecoder::new(&mut reader);
        let values: Vec<u16> = decoder.collect();
        assert_eq!(values, [0, 1, 513, u16::MAX]);
    }

    #[test]
    fn raw_u32() {
        let input: [u8; 16] = [0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 3, 4, 255, 255, 255, 255];
        let mut reader = DataReader::new(&input);
        let decoder = RawDecoder::new(&mut reader);
        let values: Vec<u32> = decoder.collect();
        assert_eq!(values, [0, 1, 67305985, u32::MAX]);
    }

    #[test]
    fn runlength_u16() {
        // Test input has three runs shown below as header; (bytes), (bytes), (...)
        //   - Two mixed values: 2; (0, 0), (0, 1)
        //   - Three of the same value: 3 | 0x80; (42, 0)
        //   - One mixed value: 1; (255, 255)
        let input: [u8; 11] = [2, 0, 0, 0, 1, 3 | 0x80, 42, 0, 1, 255, 255];
        let mut reader = DataReader::new(&input);
        let decoder = RunLengthDecoder::new(&mut reader);
        let values: Vec<u16> = decoder.collect();
        assert_eq!(values, [0, 256, 42, 42, 42, u16::MAX]);
    }

    #[test]
    fn runlength_u32() {
        // Test input has three runs shown below as header; (bytes), (bytes), (...)
        //   - Two mixed values: 2; (0, 0, 0, 0), (1, 0, 0, 0)
        //   - Three of the same value: 3 | 0x80; (42, 0, 0, 0)
        //   - One mixed value: 1; (255, 255, 255, 255)
        #[rustfmt::skip]
        let input: [u8; 19] = [
            2, 0, 0, 0, 0, 1, 0, 0, 0,
            3 | 0x80, 42, 0, 0, 0,
            1, 255, 255, 255, 255,
        ];
        let mut reader = DataReader::new(&input);
        let decoder = RunLengthDecoder::new(&mut reader);
        let values: Vec<u32> = decoder.collect();
        assert_eq!(values, [0, 1, 42, 42, 42, u32::MAX]);
    }
}

/// Necessary properties and functions for a POOL or PO32 value.
trait PoolValue<T: Copy>: Readable<T> {
    /// Wrapper around u16/u32::wrapping_add().
    fn wrapping_add(a: T, b: T) -> T;
    /// Returns 0.
    // Using the Default trait would work too...
    fn zero() -> T;

    fn to_f64(&self) -> f64;
    /// Returns 1/T::MAX as an f64, for applying scaling.
    fn recip() -> f64;
}

impl PoolValue<u32> for u32 {
    fn wrapping_add(a: u32, b: u32) -> u32 {
        a.wrapping_add(b)
    }
    fn zero() -> u32 {
        0
    }
    fn to_f64(&self) -> f64 {
        *self as f64
    }
    fn recip() -> f64 {
        1.0 / (u32::MAX as f64)
    }
}

impl PoolValue<u16> for u16 {
    fn wrapping_add(a: u16, b: u16) -> u16 {
        a.wrapping_add(b)
    }
    fn zero() -> u16 {
        0
    }
    fn to_f64(&self) -> f64 {
        *self as f64
    }
    fn recip() -> f64 {
        1.0 / (u16::MAX as f64)
    }
}

/// Represents one POOL/PO32 & SCAL/SC32 pair from a GEOD atom.
///
/// This stores the byte range from the original file of the raw POOL data with the parsed SCAL.
/// The same class is used for both POOL (16-bit) and PO32 (32-bit) types.
#[derive(Debug)]
struct Pool {
    /// The byte index within DsfInfo.data of the start of the packed pool data.
    start: usize,
    /// The byte index within DsfInfo.data of the end (exclusive) of the packed pool data.
    end: usize,

    /// The multiplier and offset for each of the planes in the pool.
    planes: Vec<(f64, f64)>,

    /// Invalid (set to usize::MAX) until the decode has been tried.
    points_per_plane: usize,

    /// The decoded values, in an interleaved plane format.
    ///
    /// All planes for each point are adjacent, so that a slice into this vector can
    /// be returned to represent a point.
    data: Vec<f64>,
}

struct PointsIter<'a> {
    pool: &'a Pool,
    pos: usize,
}

impl<'a> Iterator for PointsIter<'a> {
    type Item = &'a [f64];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.pool.num_points() {
            None
        } else {
            let result = self.pool.point(self.pos);
            self.pos += 1;
            Some(result)
        }
    }
}

impl Pool {
    fn new() -> Pool {
        Pool {
            start: 0,
            end: 0,
            planes: vec![],
            points_per_plane: usize::MAX,
            data: vec![],
        }
    }

    fn num_points(&self) -> usize {
        self.points_per_plane
    }

    fn num_planes(&self) -> usize {
        self.planes.len()
    }

    fn point(&self, i: usize) -> &[f64] {
        let np = self.num_planes();
        let start = i * np;
        &self.data[start..start + np]
    }

    fn iter(&self) -> PointsIter {
        PointsIter {
            pool: &self,
            pos: 0,
        }
    }

    fn add_plane(&mut self, multiplier: f64, offset: f64) {
        self.planes.push((multiplier, offset));
    }

    // Sets the internal start/end data; used currently only for debugging.
    fn set_offsets(&mut self, start: usize, end: usize) {
        self.start = start;
        self.end = end;
    }

    /// Parses the points from the given reader and populates the pool.
    ///
    /// Requires that add_plane() has already been called with the appropriate scaling multipliers
    /// and offsets, and that these match the number of planes in the data in the reader.
    fn decode<T>(&mut self, reader: &mut DataReader)
    where
        T: Readable<T> + Copy + PoolValue<T> + std::fmt::Debug,
    {
        self.points_per_plane = reader.read_u32() as usize;
        let plane_count: usize = reader.read_u8() as usize;
        trace!(
            "  detected {} planes with {} points per plane",
            plane_count,
            self.points_per_plane
        );
        if plane_count != self.num_planes() {
            panic!(
                "Mismatch in planes: scales are {:?} but plane_count={}",
                &self.planes, plane_count
            );
        }
        self.data.resize(self.points_per_plane * plane_count, 0.0);

        // recip is e.g. 1/65535.0 to linearly interpolate along with scale and offset.
        let recip = <T as PoolValue<T>>::recip();
        for i in 0..plane_count {
            let plane_encoding = PoolEncodingMode::from_u8(reader.read_u8());
            let scale = self.planes[i].0;
            let offset = self.planes[i].1;

            trace!(
                "  plane {} with encoding {:?} mult {} offset {}: ",
                i,
                plane_encoding,
                scale,
                offset,
            );
            let mut is_differenced = false;

            let decoder: Box<dyn Iterator<Item = T>> = match plane_encoding {
                PoolEncodingMode::Raw => Box::new(RawDecoder::new(reader)),
                PoolEncodingMode::Differenced => {
                    is_differenced = true;
                    Box::new(RawDecoder::new(reader))
                }
                PoolEncodingMode::RunLength => Box::new(RunLengthDecoder::new(reader)),
                PoolEncodingMode::RunLengthDifferenced => {
                    is_differenced = true;
                    Box::new(RunLengthDecoder::new(reader))
                }
            };

            let mut prev = <T as PoolValue<T>>::zero(); // For the differenced encodings.

            for (j, next) in decoder.take(self.points_per_plane).enumerate() {
                let idx = j * plane_count + i;
                let val: f64 = if is_differenced {
                    // Differenced mode relies on unsigned overflow wrap-around.
                    let v = <T as PoolValue<T>>::wrapping_add(next, prev);
                    prev = v;
                    v.to_f64()
                } else {
                    next.to_f64()
                };
                // Special-case: When scale is zero, no scaling is applied, per DSFTool:
                // https://github.com/X-Plane/xptools/blob/master/src/Utils/XChunkyFileUtils.cpp#L624
                let val = if scale != 0.0 {
                    val * scale * recip + offset
                } else {
                    val
                };
                self.data[idx] = val;
            }
        }
        assert!(reader.done());
        let show_points = true;
        if show_points {
            let some: Vec<&[f64]> = self.iter().take(5).collect();
            trace!("{:?}", &some);
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    #[test]
    fn u16_parsed_correctly() {
        // Primarily tests u16 parsing. Scaling and multiple planes disabled.
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            4, 0, 0, 0,  // points_per_plane: u32
            1,  // plane_count: u8
            0,  // plane 1 encoding mode: u8: Raw
            1, 0, 2, 0, 3, 0, 4, 0,  // four u16 points, raw encoded
        ];
        let mut reader = DataReader::new(&data);
        let mut pool = Pool::new();
        pool.add_plane(0.0 /* disabled */, 0.0);
        assert_eq!(1, pool.num_planes());
        pool.decode::<u16>(&mut reader);
        assert_eq!(4, pool.num_points());
        assert_eq!(&[1.0], pool.point(0));

        assert_eq!(
            vec![&[1.0], &[2.0], &[3.0], &[4.0]],
            pool.iter().collect::<Vec<&[f64]>>()
        );
    }

    #[test]
    fn u32_parsed_correctly() {
        // Primarily tests u32 parsing. Scaling and multiple planes disabled.
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            4, 0, 0, 0,  // points_per_plane: u32
            1,  // plane_count: u8
            0,  // plane 1 encoding mode: u8: Raw
            0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 3, 4, 255, 255, 255, 255, // 4 values
        ];
        let mut reader = DataReader::new(&data);
        let mut pool = Pool::new();
        pool.add_plane(0.0 /* disabled */, 0.0);
        assert_eq!(1, pool.num_planes());
        pool.decode::<u32>(&mut reader);
        assert_eq!(
            vec![&[0.0], &[1.0], &[67305985.0], &[u32::MAX as f64]],
            pool.iter().collect::<Vec<&[f64]>>()
        );
    }

    #[test]
    fn scaling_works_with_u16_single_plane() {
        // Tests primarily that scaling works. Uses u16 and these factors:
        // 65535 = 3 * 5 * 17 * 257 = 255 * 257
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            4, 0, 0, 0,  // points_per_plane: u32
            1,  // plane_count: u8
            0,  // plane 1 encoding mode: u8: Raw
            255, 0, 1, 1, 15, 0, 17, 0,  // four u16 points, raw encoded
        ];
        let mut reader = DataReader::new(&data);
        let mut pool = Pool::new();
        const SOME_MULTIPLIER: f64 = 6.0;
        const SOME_OFFSET: f64 = 100.0;
        pool.add_plane(SOME_MULTIPLIER, SOME_OFFSET);
        assert_eq!(1, pool.num_planes());
        pool.decode::<u16>(&mut reader);
        assert_eq!(
            vec![
                &[100.0 + 6.0 / 257.0],
                &[100.0 + 6.0 / 255.0],
                &[100.0 + 6.0 / 4369.0],
                &[100.0 + 6.0 / 3855.0]
            ],
            pool.iter().collect::<Vec<&[f64]>>()
        );
    }

    #[test]
    fn scale_factors_applied_with_multiple_pools() {
        // Tests that scaling factors line up with each pool.
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            2, 0, 0, 0,  // points_per_plane: u32
            3,  // plane_count: u8
            0,  // plane 1 encoding mode: u8: Raw
            1, 0, 2, 0,
            0,  // plane 2 encoding mode: Raw
            3, 0, 4, 0,
            0,  // plane 3 encoding mode: Raw
            5, 0, 6, 0,
        ];
        let mut reader = DataReader::new(&data);
        let mut pool = Pool::new();
        const MULT: f64 = 6.0;
        const OFFSET: f64 = 100.0;
        pool.add_plane(0.0 /* disabled */, 0.0);
        pool.add_plane(MULT, OFFSET);
        pool.add_plane(0.0 /* disabled */, 0.0);
        assert_eq!(3, pool.num_planes());
        pool.decode::<u16>(&mut reader);
        assert_eq!(
            vec![
                &[1.0, OFFSET + MULT * 3.0 * (1.0 / u16::MAX as f64), 5.0],
                &[2.0, OFFSET + MULT * 4.0 * (1.0 / u16::MAX as f64), 6.0],
            ],
            pool.iter().collect::<Vec<&[f64]>>()
        );
    }

    #[test]
    fn pool_encoding_modes() {
        #[rustfmt::skip]
        let data: Vec<u8> = vec![
            4, 0, 0, 0,  // points_per_plane: u32
            3,  // plane_count: u8
            1,  // plane 1 encoding mode: u8: Differenced
            1, 0, 0, 0, 10, 0, 100, 0,
            2,  // plane 2 encoding mode: RunLength (4@42)
            4 | 0x80, 42, 0,
            3,  // plane 3 encoding mode: RunLengthDifferenced (4@+2)
            4 | 0x80, 2, 0,
        ];
        let mut reader = DataReader::new(&data);
        let mut pool = Pool::new();
        pool.add_plane(0.0 /* disabled */, 0.0);
        pool.add_plane(0.0 /* disabled */, 0.0);
        pool.add_plane(0.0 /* disabled */, 0.0);
        assert_eq!(3, pool.num_planes());
        pool.decode::<u16>(&mut reader);
        assert_eq!(
            vec![
                &[1.0, 42.0, 2.0],
                &[1.0, 42.0, 4.0],
                &[11.0, 42.0, 6.0],
                &[111.0, 42.0, 8.0],
            ],
            pool.iter().collect::<Vec<&[f64]>>()
        );
    }
}

// Pool Value Encoding
//
// - Each pool has one or more planes.
//   - Each plane has the same number of points.
//   - There is exactly one scale atom per pool, which transforms the pool vals into global vals.
// - A pool's raw values (after decompression) may be 16 or 32 bits wide.
// - The raw values within a pool may be encoded directly and/or by delta and run-length encoding.
//
// The decoded numbers are stored interleaved in memory (i*width+plane) so that all related
// values are adjacent. (The reference implementation supports both but it appears the
// non-interleaved path is dead code.)

/// This describes the way that the data in pools is encoded.
///
#[derive(Debug)]
enum PoolEncodingMode {
    /// Values are encoded as themselves, in little-endian byte order.
    Raw = 0,

    /// Values are the sum of the previous value and the currently encoded one.
    ///
    /// The values are interpreted pre-scaling (i.e. over the 16- or 32-bit unsigned data
    /// in the pool), and the encoding relies on overflow to encode negative deltas.
    ///
    /// Note that this encoding method doesn't save any space relative to Raw, unless some outer
    /// compression takes advantage of the resulting bytes.
    Differenced = 1,

    /// Values are encoded in a run-length format.
    ///
    /// Each run starts with a one-byte header. The lower 7 bits indicate the number of values
    /// in the run, up to 127. If the 0x80 bit is set, then the header is a run of the same value,
    /// and that value follows in the next 2 or 4 bytes. Otherwise, N values follow in the next
    /// 2*N or 4*N bytes. If there is more data, another block of header and data follows.
    RunLength = 2,

    /// The values are RunLength encoded Differences, as described in the two methods above.
    ///
    /// This method is efficient at encoding many raw values that are either identical (with
    /// differences of zero) or evenly-spaced, because the resulting runs of 0, 1, etc.,
    /// compress well in a RunLength encoding.  It seems to be a very common format (in 2022).
    RunLengthDifferenced = 3,
}

impl PoolEncodingMode {
    fn from_u8(value: u8) -> PoolEncodingMode {
        match value {
            0 => PoolEncodingMode::Raw,
            1 => PoolEncodingMode::Differenced,
            2 => PoolEncodingMode::RunLength,
            3 => PoolEncodingMode::RunLengthDifferenced,
            _ => panic!("Encountered unknown value {} for PoolEncodingMode.", value),
        }
    }
}

/// Constants for each of the Atom types found in a DSF file.
///
/// Descriptions and information taken from:
/// <https://developer.x-plane.com/article/dsf-file-format-specification/>
/// <https://developer.x-plane.com/article/dsf-usage-in-x-plane/>
///
/// They are reproduced here because the information on these pages (and the pages themselves)
/// has moved around in the past.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AtomId {
    /// Header Atom (HEAD)
    ///
    /// This is an atom of atoms containing information about the DSF file. The HEAD atom is an
    /// atom-of-atoms in the top level of the file and currently contains one subatom the PROP
    /// atom.
    HEAD = 0x48454144,

    /// Definitions Atom (‘DEFN’)
    ///
    /// The definitions atom contains a series of subatoms that define the various ‘definitions’
    /// used within the DSF file. DSF files first reference a few common definitions and then use
    /// them in a larger number of instances. Each definition comes from an external file, allowing
    /// definitions to be shared among DSF files or even between scenery packages. (This allows
    /// custom scenery packages to use a variety of x-plane-default definitions.) The various
    /// definition formats are described in other specifications.
    ///
    /// The definition atom is an atom-of-atoms syntactically. All definition sub atoms define a
    /// series of partial file paths by being string table atoms. The forward slash (‘/’) should be
    /// used as the directory separator. All definitions are referred to by zero-based index in the
    /// rest of the file. A maximum of 65536 entries are allowed in any one table. The extension
    /// for the file path is always included.
    ///
    /// The following four atoms sit inside the definitions atom: the TERT, OBJT, POLY and NETW
    /// atoms.
    DEFN = 0x4445464e,

    /// Geodata Atom (‘GEOD’)
    ///
    /// The geodata atom defines all of the coordinates for all geometry in the DSF file.
    /// Coordinates are separated from instantiations of definitions to encourage recycling and
    /// reduce file size.
    ///
    /// The Geodata atom is an atom-of-atoms at the root of the DSF file. The GEOD atom contains
    /// zero or more POOL, SCAL, PO32 and SC32 atoms.
    GEOD = 0x47454f44,

    /// Commands Atom (‘CMDS’)
    ///
    /// The commands atom contains a list of commands used to actually instantiate the scenery file
    /// by applying prototypes, objects, etc. at the coordinates available in the geodata atom.
    ///
    /// Commands consist of a command ID and additional information in series. Command order is
    /// arbitrary but may be optimized for file size by the file writer. Command order does not
    /// affect display order when X-Plane renders scenery; display order is affected by internal
    /// factors in the rendering engine.
    ///
    /// The number of bytes used by a command is known through its type; unknown commands cannot be
    /// skipped. The commands are finished when the last command in the atom is parsed. All command
    /// IDs are 8-bit. Command ID 255 is reserved for future expansion. The format of the data
    /// following the command is based on the command ID.
    ///
    /// All commands that include a range of indices list the first index, and one more than the
    /// last index. All commands reference 16-bit point pools except for vector commands, which
    /// reference 32-bit point pools.
    CMDS = 0x434d4453,

    /// Raster Data Atom (‘DEMS’)
    ///
    /// New to X-Plane 10: The raster data atom is an atom of atoms containing the meta data and
    /// raw data for each raster layer in the DSF.
    ///
    /// The raster data atom contains one raster layer information (‘DEMI’) and one raster layer
    /// data atom (‘DEMD’) for each raster layer. The order of information and data atoms must
    /// match with the raster definition atoms. This is how names, meta data, and raw data are
    /// matched in the DSF.
    DEMS = 0x44454d53,

    // ============================== HEAD sub-atoms ==============================//
    /// Properties Atom (PROP)
    ///
    /// The properties atom is a string table atom with an even number of strings; each consecutive
    /// pair of strings represents a property name and a property value. This allows for arbitrary
    /// metadata to be placed inside a DSF file.
    ///
    /// Properties starting with sim/ are reserved for public X-Plane use. Properties starting with
    /// laminar/ are reserved for X-Plane private use. All other prefixes may be used for private
    /// data. When storing private data in the DSF properties metadata section, prefix the property
    /// with your company or organization name to prevent conflicts.
    ///
    /// Property	Default If Missing	Definition
    /// sim/west	(Required)	The western edge of the DSF file in degrees longitude.
    /// sim/east	(Required)	The eastern edge of the DSF file in degrees longitude.
    /// sim/south	(Required)	The northern edge of the DSF file in degrees latitude.
    /// sim/north	(Required)	The southern edge of the DSF file in degrees latitude.
    /// sim/planet	earth	The planet this DSF belongs to, one of ‘earth’ or ‘mars’.
    /// sim/creation_agent	(Blank)	The name of the program that created the DSF file if known.
    /// sim/author	(Blank)	The name of the author of the DSF file if known.
    /// sim/require_object	N/A	Requirements for displaying objects (see below).
    /// sim/require_facade	N/A	Requirements for displaying facades (see below).
    ///
    /// The sim/require_object and sim/require_facade properties specify that objects and facade
    /// whose definition index is greater than or equal to a certain number must be drawn by the
    /// sim. Normally the sim may draw only a fraction of the objects or facades in a DSF, based on
    /// the user’s rendering settings. By including these properties, you can force X-Plane to
    /// always draw objects.
    PROP = 0x50524f50,

    // ============================== DEFN sub-atoms ==============================//
    /// TERRAIN TYPES ATOM (‘TERT’)
    ///
    /// The terrain types atom lists a number of external .ter terrain-definition files that define
    /// the various terrains used in the DSF file. Terrain-definition files describe the set of
    /// textures to be used for the terrain (dependent on season) as well as other metadata (for
    /// example, is this terrain hard, bumpy, etc.).
    ///
    /// You may also use .png or .bmp files directly to specify terrain types. See the .ter file
    /// specification for info on both the .ter file format and on using PNG and BMP files here.
    TERT = 0x54455254,

    /// OBJECTS ATOM (‘OBJT’)
    ///
    /// The objects atom lists a number of external .obj object files that may be ‘placed’
    /// repeatedly in the DSF file. With DSF files there are no default object types; radio stacks,
    /// sky scrapers, etc. are all created using either Object or Prototype (see below) files.
    OBJT = 0x4f424a54,

    /// THE POLYGONS ATOM (‘POLY’)
    ///
    /// The prototype atom lists a number of external polygon definition files that may be usde
    /// within the DSF file.
    ///
    /// Polygon definitions can be either facades or forests; X-Plane determines the type of
    /// polygon definition from the filename extension.
    POLY = 0x504f4c59,

    /// VECTOR NETWORK ATOM (‘NETW’)
    ///
    /// The network atom lists a number of external .net network definition files that may be used
    /// within the DSF file. While individual objects are placed separately in a file, roads and
    /// other ‘networks’ intersect each other. A network definition file describes the appearance
    /// and geometry not only of multiple different types of roads (or other segments), but how to
    /// build blended intersections of those segments.
    ///
    /// X-PLANE NOTE: X-Plane 8 and 9 can only accept one network definition per DSF file; use
    /// vector subtypes to define multiple road types, etc.
    NETW = 0x4e455457,

    /// RASTER DEFINITION ATOM (‘DEMN’)
    ///
    /// New to X-Plane 10: The raster definition atom defines the names for each raster layer
    /// contained in the DSF. The order of raster data in the subsequent atoms matches the order of
    /// names in the raster definition atom.
    DEMN = 0x44454d4e,

    // ============================== GEOD sub-atoms ==============================//
    /// 16-BIT COORDINATE POOL ATOM (‘POOL’)
    ///
    /// This atom is a planar numeric atom with a variable number of planes and 16-bit unsigned int
    /// data, establishing a coordinate pool. Multiple pool atoms sit inside the Geodata atom, so
    /// the index number of this coordinate pool is based on its order within the geodata atom,
    /// starting at 0. Points are stored in sixteen bit unsigned short format, representing values
    /// from [0-65536).
    POOL = 0x504f4f4c,

    /// 16-BIT SCALING RANGE ATOMS (‘SCAL’)
    ///
    /// For each pool atom there is also a scaling range atom in the GeoData atom, telling how to
    /// process the point pools. Each scaling atom contains an array of 32-bit floating point
    /// numbers. There are two floats for each plane in the corresponding point pool, the first
    /// being a scaling multiplier and the second being an offset to be added to the points. These
    /// values are applied in double-precision.
    ///
    /// POOL and SCAL atoms are applied based on their order within the file, e.g. the 5th POOL
    /// atom within the GEOD atom is scaled by the 5th SCAL atom in the GEOD atom. There must be an
    /// equal number of POOL and SCAL atoms. The data planes in the pool atom correspond to the
    /// values in the scal atom, so if there are n planes in a POOL atom, its corresponding SCAL
    /// atom must have 2n 32-bit floats.
    SCAL = 0x5343414c,
    /// 32-BIT COORDINATE POOL ATOM (‘PO32’)
    ///
    /// The 32-bit point pool atom is the same as the 16-bit point pool atom except that each data
    /// element is a 32-bit rather than 16-bit unsigned int. 32-bit point pool atoms are used for
    /// vectors; all other building blocks use 16-bit point pools.
    PO32 = 0x504f3332,

    /// 32-BIT SCALING RANGE ATOM (‘SC32’)
    ///
    /// The 32-bit scaling range atom is the same as the 16-bit scaling range atom except that it
    /// is appleid to 32-bit point pool atoms. In other words, the 3rd SC32 atom scales the 3rd
    /// PO32 atom. The atom is still formed of 32-bit floats, but like the 16-bit scaling atom, the
    /// conversion is done in double-precision floating point.
    SC32 = 0x53433332,

    // ============================== DEMS sub-atoms ==============================//
    /// RASTER LAYER INFORMATION ATOM (‘DEMI’)
    ///
    /// The raster layer information atom contains the structure information for one raster layer.
    /// The atom is a record of DEM information, encoded as follows:
    ///
    /// Field	Encoding	Description
    /// Version	uint8	Version of DEM record; set to 1
    /// Bytes Per Pixel	uint8	The number of bytes for each pixel. Should be 1,2, or 4 depending on encoding
    /// Flags	uint16	Encoding Flags – see below
    /// Width	uint32	Width of the DEM east-west in pixels
    /// Height	uint32	Height of the DEM north-south in pixels
    /// Scale	float32	Scaling factor to apply to DEM pixels post-load
    /// Offset	float32	Offset factor to apply to DEM pixels post-load
    /// Each final DEM pixel is multiplied by scale and then offset is added.
    ///
    /// The flags field defines a number of other DEM properties:
    ///
    /// The low 2 bits tell the number type for the DEM data:
    /// 0 = floating point (bytes per pixel must be 4)
    /// 1 = signed integer (bytes per pixel must be 1, 2 or 4)
    /// 2 = unsigned integer (bytes per pixel must be 1, 2 or 4)
    /// A flag value of 4 (bit 3) defines the data as post-centric, as opposed to area-centric.
    ///
    /// In post-centric data, the pixel values at the edges of the DEM exactly lie on the geometric
    /// boundary of the DSF.  In point-centric data, the outer edge of the pixel rectangles lie on
    /// the geometric boundary of the DSF.
    DEMI = 0x44454d49,

    /// RASTER LAYER DATA ATOM (‘DEMD’)
    ///
    /// The raster data atom contains the actual raw raster data, sitting directly in the atom’s
    /// payload, one DEMD atom per layer. The information atom above tells how to inerpret this raw
    /// data.
    DEMD = 0x44454d44,
}

impl AtomId {
    /// Parses value as an AtomId. Crashes if value is invalid.
    fn from_i32(value: i32) -> AtomId {
        match value {
            0x48454144 => AtomId::HEAD,
            0x4445464e => AtomId::DEFN,
            0x47454f44 => AtomId::GEOD,
            0x434d4453 => AtomId::CMDS,
            0x44454d53 => AtomId::DEMS,

            0x50524f50 => AtomId::PROP,

            0x54455254 => AtomId::TERT,
            0x4f424a54 => AtomId::OBJT,
            0x504f4c59 => AtomId::POLY,
            0x4e455457 => AtomId::NETW,
            0x44454d4e => AtomId::DEMN,

            0x504f4f4c => AtomId::POOL,
            0x5343414c => AtomId::SCAL,
            0x504f3332 => AtomId::PO32,
            0x53433332 => AtomId::SC32,

            0x44454d49 => AtomId::DEMI,
            0x44454d44 => AtomId::DEMD,
            _ => panic!("Encountered unknown value {} for AtomId.", value),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum CommandId {
    /// Reserved value, which should never appear.
    Reserved = 0,
    /// Changes the current coordinate pool and estables the lat/long bounds.
    PoolSelect = 1,
    /// Specifies a 32-bit number that is added to all indices when referencing vector coords.
    ///
    /// This allows the use of a 16-bit vector command for vectors whose indices are greater
    /// than 65535.
    JunctionOffsetSelect = 2,
    /// ?
    SetDefinition8 = 3,
    /// ?
    SetDefinition16 = 4,
    /// ?
    SetDefinition32 = 5,
    /// Sets the road subtype for the next vector-segment.
    SetRoadSubtype8 = 6,
    /// Places an object on the mesh surface.
    ///
    /// A point pool must be selected and have at least 3 planes, which are treated as a
    /// longitude, latitude, and rotation in degrees.
    Object = 7,
    /// Like Object but places objects at all points in a [start_idx, end_idx) range.
    ObjectRange = 8,
    /// Creates one or more Network Chains, using all the vertex indicies given.
    NetworkChain = 9,
    /// Creates one or more Network Chains, using all the verticies in [start_idx, end_idx).
    NetworkChainRange = 10,
    /// Creates one or more Network Chains, using the given 32-bit indices.
    NetworkChain32 = 11,
    /// Places one or more polygon primitives on the surface of the mesh.
    ///
    /// The selected plane must have at least two planes, which are interpreted as longitude
    /// and latitude. A per-polygon 16-bit parameter is interpreted based on the polygon
    /// definition.
    ///
    /// Arguments are the parameter, the count, and a list of count 16-bit indices.
    Polygon = 12,
    /// Places polygons on all verticies in a [start, end) range.
    PolygonRange = 13,
    /// Places a series of polygons, each with a distinct winding.
    NestedPolygon = 14,
    /// Places a series of polygons with distinct windings using a list of contiguous ranges.
    NestedPolygonRange = 15,
    /// Indicates that a new terrain patch is being created.
    ///
    /// The patch will have the same LOD range and flags that the last created patch had.
    TerrainPatch = 16,
    /// Indicates that a new terrain patch is being created, with new flags (and same LOD range).
    ///
    /// The flags are:
    ///   0x1: Physical: if set, this patch is used for collision detection.
    ///   0x2: Overlay: if set, this patch is drawn over another patch.
    TerrainPatchFlags = 17,
    /// Indicates that a new terrain patch is being created, with the given flags and LOD range.
    ///
    /// Parameters are the flags as listed above, and the new LOD range [near, far) in meters.
    TerrainPatchFlagsLOD = 18,

    // 19 - 22 are not used.
    /// Creates one or more specific triangles for a terrain patch.
    Triangle = 23,
    /// Creates one triangle from multiple terrain pools.
    ///
    /// This is the same as Triangle, but one pool index is provided per vertex.
    TriangleCrossPool = 24,
    /// Creates a number of triangles based on the range of vertices [start, end).
    ///
    /// Each set of 3 adjacent vertices is treated as a triangle.
    TriangleRange = 25,
    /// Creates a triangle strip.
    ///
    /// A triangle strip is a series of adjacent triangles that share two common vertices; for
    /// a series of points 1,2,3,4,5 as a triangle strip is equivalent to the triangles 123, 234,
    /// 345, ...
    TriangleStrip = 26,
    /// Creates a triangle strip, except the point pool is specified per vertex rather than
    /// referencing the current pool.
    TriangleStripCrossPool = 27,
    /// Creates a triangle strip from a series of consecutive coordinates.
    TriangleStripRange = 28,
    /// Creates a triangle fan.
    ///
    /// A triangle fan is a series of adjacent triangles that share two common vertices; for a
    /// series of points 1,2,3,4,5 as a triangle fan is equivalent to 123, 134, 145, ...
    TriangleFan = 29,
    /// Creates a triangle fan with the point pools specified per vertex.
    TriangleFanCrossPool = 30,
    /// Creates a triangle fan from a series of consecutive coordinates.
    TriangleFanRange = 31,
    /// Embeds an arbitrary comment up to 255 characters long.
    Comment8 = 32,
    /// Embeds an arbitrary comment up to 65535 characters long.
    Comment16 = 33,
    /// Embeds an arbitrary comment up to 2^32 characters long.
    Comment32 = 34,
}

impl CommandId {
    fn from_u8(value: u8) -> CommandId {
        match value {
            0 => CommandId::Reserved,
            1 => CommandId::PoolSelect,
            2 => CommandId::JunctionOffsetSelect,
            3 => CommandId::SetDefinition8,
            4 => CommandId::SetDefinition16,
            5 => CommandId::SetDefinition32,
            6 => CommandId::SetRoadSubtype8,
            7 => CommandId::Object,
            8 => CommandId::ObjectRange,
            9 => CommandId::NetworkChain,
            10 => CommandId::NetworkChainRange,
            11 => CommandId::NetworkChain32,
            12 => CommandId::Polygon,
            13 => CommandId::PolygonRange,
            14 => CommandId::NestedPolygon,
            15 => CommandId::NestedPolygonRange,
            16 => CommandId::TerrainPatch,
            17 => CommandId::TerrainPatchFlags,
            18 => CommandId::TerrainPatchFlagsLOD,

            // 19 - 22 are not used.
            23 => CommandId::Triangle,
            24 => CommandId::TriangleCrossPool,
            25 => CommandId::TriangleRange,
            26 => CommandId::TriangleStrip,
            27 => CommandId::TriangleStripCrossPool,
            28 => CommandId::TriangleStripRange,
            29 => CommandId::TriangleFan,
            30 => CommandId::TriangleFanCrossPool,
            31 => CommandId::TriangleFanRange,
            32 => CommandId::Comment8,
            33 => CommandId::Comment16,
            34 => CommandId::Comment32,
            _ => panic!("Encountered unknown command {}", value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ElevMode {
    AGL,
    MSL,
    Draped,
}

/// Describes a type that can be used with DataReader::read().
///
/// This trampoline/whatnot is to allow DataReader to be called from functions that
/// use generic types (and want to parse different types accordingly).
// There's probably a better way to do this, but this works for now.
trait Readable<T: Copy> {
    /// Reads a T as little-endian bytes from the start of reader, and advances the
    /// reader accordingly.
    fn read(reader: &mut DataReader) -> T;
}

impl Readable<u32> for u32 {
    fn read(reader: &mut DataReader) -> u32 {
        reader.read_u32()
    }
}

impl Readable<u16> for u16 {
    fn read(reader: &mut DataReader) -> u16 {
        reader.read_u16()
    }
}

/// This is a wrapper around a byte slice that keeps a position and enables easy parsing.
struct DataReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl DataReader<'_> {
    /// Creates a reader over the given slice and sets the position to the beginning.
    fn new(data: &[u8]) -> DataReader {
        DataReader { data: data, pos: 0 }
    }

    fn read_u8(&mut self) -> u8 {
        let result = self.data[self.pos];
        self.pos += 1;
        assert!(self.pos <= self.data.len());
        return result;
    }

    fn read_u16(&mut self) -> u16 {
        let result = self.data[self.pos] as u16 | ((self.data[self.pos + 1] as u16) << 8);
        self.pos += 2;
        assert!(self.pos <= self.data.len());
        return result;
    }

    fn read_i16(&mut self) -> i16 {
        let input: [u8; 2] = [self.data[self.pos], self.data[self.pos + 1]];
        self.pos += 2;
        i16::from_le_bytes(input)
    }

    fn read_u32(&mut self) -> u32 {
        let result = self.data[self.pos] as u32
            | ((self.data[self.pos + 1] as u32) << 8)
            | ((self.data[self.pos + 2] as u32) << 16)
            | ((self.data[self.pos + 3] as u32) << 24);
        self.pos += 4;
        assert!(self.pos <= self.data.len());
        return result;
    }

    fn read_i32(&mut self) -> i32 {
        let result = self.data[self.pos] as i32
            | ((self.data[self.pos + 1] as i32) << 8)
            | ((self.data[self.pos + 2] as i32) << 16)
            | ((self.data[self.pos + 3] as i32) << 24);
        self.pos += 4;
        assert!(self.pos <= self.data.len());
        return result;
    }

    fn read_f32(&mut self) -> f32 {
        let input: [u8; 4] = [
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ];
        self.pos += 4;
        assert!(self.pos <= self.data.len());
        f32::from_le_bytes(input)
    }

    fn read<T: Readable<T> + Copy>(&mut self) -> T {
        T::read(self)
    }

    fn skip_n_bytes(&mut self, n: usize) {
        self.pos += n;
        assert!(self.pos <= self.data.len());
    }

    fn done(&self) -> bool {
        return self.pos >= self.data.len();
    }
}

/// A DSF StringTable containing a list of strings identified by index starting at 0.
#[derive(Debug, Default)]
struct StringTable {
    /// The strings from the table, in file order.
    strings: Vec<String>,

    /// (optional) The number of commands/etc. that reference each string.
    /// This doesn't make sense for the PROP table, but it does for the rest.
    file_refs: Vec<usize>,

    /// (optional) The number of instantiations of each string.
    /// This doesn't make sense for the PROP table, but it does for the rest.
    object_refs: Vec<usize>,
}

impl fmt::Display for StringTable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "String table with {} entries:", self.len())?;
        for (i, s) in self.strings.iter().enumerate() {
            writeln!(f, "  {}: {}\t\t[{}]", i, &s, self.object_refs[i])?;
        }
        Ok(())
    }
}

impl StringTable {
    /// Creates a new StringTable or throws an error.
    ///
    /// Requires that the strings in the StringTable are UTF-8 clean.
    fn new(data: &[u8], _global_offset: usize) -> Result<StringTable, Box<dyn Error>> {
        let mut table: StringTable = Default::default();
        table.init(data)?;
        Ok(table)
    }

    /// Returns how many strings are in the table.
    fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns whether the table is empty.
    fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Returns the i-th string.
    fn get(&self, i: usize) -> &str {
        &self.strings[i]
    }

    /// Bumps all the references in the given vector.
    fn add_all_refs(&mut self, refs: &Vec<usize>) {
        for (i, v) in refs.iter().enumerate() {
            self.file_refs[i] += v;
            self.object_refs[i] += v;
        }
    }

    /// Initializes the internal map of string positions, and in doing so checks for UTF-8.
    fn init(&mut self, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let mut pos = 0;
        let mut last_start = 0;
        assert!(self.strings.is_empty()); // Don't call init() multiple times!
        while pos < data.len() {
            if data[pos] == 0 {
                // NUL
                let s = std::str::from_utf8(&data[last_start..pos])?; // Or error.
                self.push(s);
                last_start = pos + 1;
            }
            pos += 1;
        }
        Ok(())
    }

    /// Adds the given string to the end of the table.
    fn push(&mut self, s: &str) {
        self.strings.push(String::from(s));
        self.file_refs.push(0);
        self.object_refs.push(0);
    }

    /// Creates a dummy StringTable for use in tests and placeholders.
    fn dummy() -> StringTable {
        Default::default()
    }
}

/// Represents the contents of the PROP / Properties atom.
#[derive(Debug)]
struct PropertiesAtom {
    /// The raw table.
    table: StringTable,

    /// The extracted / cached bounds, as (west, south, east, north) degrees.
    bounds: (i32, i32, i32, i32),

    /// Whether `sim/overlay` was present and set to 1.
    is_overlay: bool,

    /// The number of exclusion entries found in the properties.
    num_exclusions: usize,

    /// The kinds of exclusion entries found in the properties, with their counts.
    exclusion_kinds: HashMap<String, usize>,
}

impl fmt::Display for PropertiesAtom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Properties:")?;
        for (k, v) in self.pairs() {
            writeln!(f, "  {}: {}", k, v)?
        }
        Ok(())
    }
}

impl PropertiesAtom {
    fn new(table: StringTable) -> Result<PropertiesAtom, Box<dyn Error>> {
        if table.strings.len() % 2 == 1 {
            bail!("Tried to initialize PropertiesAtom with an odd number of strings.");
        }
        let mut out = PropertiesAtom {
            table: StringTable::dummy(),
            bounds: (-1000, -1000, -1000, -1000),
            is_overlay: false,
            num_exclusions: 0,
            exclusion_kinds: HashMap::<String, usize>::new(),
        };

        let mut iter = table.strings.iter();
        loop {
            let key = iter.next();
            if key.is_none() {
                break;
            }
            let key = key.unwrap();
            let value = iter.next().unwrap();
            // REQUIRED PROPERTIES
            if key == "sim/west" {
                out.bounds.0 = value.parse()?;
            } else if key == "sim/south" {
                out.bounds.1 = value.parse()?;
            } else if key == "sim/east" {
                out.bounds.2 = value.parse()?;
            } else if key == "sim/north" {
                out.bounds.3 = value.parse()?;
            } else if key == "sim/planet" {
                if value != "earth" {
                    bail!(format!("Don't know how to handle non-Earth planets."));
                }

            // OVERLAY PROPERTIES
            } else if key == "sim/overlay" {
                if value == "1" {
                    out.is_overlay = true;
                } else if value != "0" {
                    bail!(format!("Unexpected value for sim/overlay of {}", &value));
                }
            } else if key.starts_with("sim/exclude_") {
                let kind = String::from(&key[12..]);
                let counter = out.exclusion_kinds.entry(kind).or_insert(0);
                *counter += 1;
                out.num_exclusions += 1;

            // OBJECT_DENSITY PROPERTIES
            } else if key == "sim/require_object"
                || key == "sim/require_agp"
                || key == "sim/require_agpoint"
                || key == "sim/require_facade"
            {
                // These adjust object density. Skip.
                // LOD PROPERTIES
            } else if key == "sim/lod_mesh" {
                // Skip.
                // OTHER PROPERTIES
            } else if key.starts_with("laminar/") {
                // Skip
            } else if key == "sim/creation_agent" || key == "sim/internal_revision" {
                info!("DSF {} = {}", &key, &value);
            } else {
                warn!("DSF: Unknown property {} = {}", &key, &value);
            }
        }
        if out.bounds.0 < -180
            || out.bounds.0 > 180
            || out.bounds.1 < -90
            || out.bounds.1 > 90
            || out.bounds.2 < -180
            || out.bounds.2 > 180
            || out.bounds.3 < -90
            || out.bounds.3 > 90
        {
            bail!(format!("Invalid bounds in DSF: {:?}", &out.bounds));
        }

        out.table = table;
        Ok(out)
    }

    /// Returns true if the sim/overlay=1 property is set.
    fn is_overlay(&self) -> bool {
        return self.is_overlay;
    }

    /// Returns the number of exclusion entries found in the DSF's properties.
    fn num_exclusions(&self) -> usize {
        return self.num_exclusions;
    }

    /// Returns the value of the given key or the empty string if not set.
    fn get_or_empty(&self, key: &str) -> String {
        for (k, v) in self.pairs() {
            if k == key {
                return v.clone();
            }
        }
        return String::new();
    }

    /// Returns an iterator over the key-value pairs in the table.
    fn pairs(&self) -> impl Iterator<Item = (&String, &String)> {
        use itertools::Itertools;
        self.table.strings.iter().tuples::<(&String, &String)>()
    }
}

/// Represents a raster header and optionally the corresponding data.
#[derive(Debug, Default)]
pub struct RasterData {
    /// Valid values are 1, 2, or 4.
    bytes_per_pixel: u8,

    /// The lower two bits indicate data type (0=float, 1=int, 2=unsigned). 0x4 is the centric flag:
    ///
    /// "A flag value of 4 (bit 3) defines the data as post-centric, as opposed to area-centric. In
    /// post-centric data, the pixel values at the edges of the DEM exactly lie on the geometric
    /// boundary of the DSF.  In point-centric data, the outer edge of the pixel rectangles lie on
    /// the geometric boundary of the DSF."
    flags: u16,

    width: u32,
    height: u32,
    scale: f32,
    offset: f32,

    data: Vec<u8>,
}

impl fmt::Display for RasterData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let info = match (self.flags & 0x3, self.bytes_per_pixel) {
            (0, 4) => "f32",
            (1, 1) => "i8",
            (1, 2) => "i16",
            (1, 4) => "i32",
            (2, 1) => "u8",
            (2, 2) => "u16",
            (2, 4) => "u32",
            _ => panic!(
                "Invalid DEMI flags and bytes per pixel: {}, {}",
                self.flags, self.bytes_per_pixel
            ),
        };
        write!(
            f,
            "DEMI information: {}x{} {} pixels with scale={}, offset={}; post_centric={}",
            self.width,
            self.height,
            info,
            self.scale,
            self.offset,
            self.is_post_centric()
        )
    }
}

impl RasterData {
    pub fn is_post_centric(&self) -> bool {
        self.flags & 0x4 != 0
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn bounds_unscaled(&self) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for i in 0..self.width() {
            for j in 0..self.height() {
                let v = self.unscaled(i, j);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        return (min, max);
    }

    /// Returns the unscaled value at the given x,y point, converted to f32 if necessary.
    // x and y are u32 to match the limitations of the underlying type
    pub fn unscaled(&self, x: u32, y: u32) -> f32 {
        let index: usize = (y * self.width() + x) as usize;
        let pos = index * (self.bytes_per_pixel as usize);
        if self.bytes_per_pixel == 4 {
            let raw = [
                self.data[pos],
                self.data[pos + 1],
                self.data[pos + 2],
                self.data[pos + 3],
            ];
            match self.flags & 0x3 {
                0 => f32::from_le_bytes(raw),
                1 => i32::from_le_bytes(raw) as f32,
                2 => u32::from_le_bytes(raw) as f32,
                _ => panic!("invalid format"),
            }
        } else if self.bytes_per_pixel == 2 {
            let raw = [self.data[pos], self.data[pos + 1]];
            match self.flags & 0x3 {
                1 => i16::from_le_bytes(raw) as f32,
                2 => u16::from_le_bytes(raw) as f32,
                _ => panic!("invalid format"),
            }
        } else {
            assert_eq!(1, self.bytes_per_pixel);
            let raw = [self.data[pos]];
            match self.flags & 0x3 {
                1 => i8::from_le_bytes(raw) as f32,
                2 => u8::from_le_bytes(raw) as f32,
                _ => panic!("invalid format"),
            }
        }
    }

    /// Returns the scaled value at the given x,y point.
    pub fn data(&self, x: u32, y: u32) -> f32 {
        self.unscaled(x, y) * self.scale + self.offset
    }
}

/// Represents one DSF file.
#[derive(Debug, Default)]
pub struct DsfInfo {
    /// The filesystem path used to access the file.
    path: PathBuf,

    /// Whether the original file was compressed or not.
    compressed: bool,

    /// The raw file bytes of the entire file.
    data: Vec<u8>,

    /// The contents of the PROP atom.
    properties: Option<PropertiesAtom>,

    /// The contents of the TERT atom.
    terrain_table: StringTable,

    /// The contents of the OBJT atom.
    object_table: StringTable,

    /// The contents of the POLY atom.
    poly_table: StringTable,

    /// The contents of the NETW atom.
    network_table: StringTable,

    /// The contents of the DEMN atom.
    raster_definitions: StringTable,

    /// The POOL+SCAL definitions.
    pools: Vec<Pool>,

    /// The PO32+SC32 definitions.
    po32s: Vec<Pool>,

    /// Raster data.
    rasters: Vec<RasterData>,
}

impl fmt::Display for DsfInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_tables = f.sign_plus();
        writeln!(f, "DSF: {}", self.path.display())?;
        writeln!(
            f,
            "    size {} bytes; original {}",
            self.data.len(),
            if self.compressed {
                "compressed"
            } else {
                "uncompressed"
            }
        )?;
        let props = self.properties.as_ref().unwrap();
        writeln!(
            f,
            "    W,S,E,N: {:?}; overlay={}; num_exclusions={}",
            props.bounds, props.is_overlay, props.num_exclusions
        )?;
        let exclusion_counts: Vec<String> = props
            .exclusion_kinds
            .iter()
            .map(|(k, &v)| format!("{}={}", &k, &v))
            .collect();
        writeln!(f, "    exclusions: {}", exclusion_counts.join(", "))?;

        let terrain = &self.terrain_table;
        let objects = &self.object_table;
        let polys = &self.poly_table;
        let networks = &self.network_table;
        let rasters = &self.raster_definitions;

        writeln!(f, "    terrain:    {} items", terrain.len())?;
        writeln!(f, "    objects:    {} items", objects.len())?;
        writeln!(f, "    poly:       {} items", polys.len())?;
        writeln!(f, "    networks:   {} items", networks.len())?;
        writeln!(f, "    rasters:    {} items", rasters.len())?;
        writeln!(
            f,
            "    pools:      {} short, {} long",
            self.pools.len(),
            self.po32s.len()
        )?;
        if print_tables {
            writeln!(f, "\n{}", props)?;
            if !terrain.is_empty() {
                writeln!(f, "Terrain Table:\n{}", terrain)?;
            }
            if !objects.is_empty() {
                writeln!(f, "Object Table:\n{}", objects)?;
            }
            if !polys.is_empty() {
                writeln!(f, "Poly Table:\n{}", polys)?;
            }
            if !networks.is_empty() {
                writeln!(f, "Network Table:\n{}", networks)?;
            }
            if !rasters.is_empty() {
                writeln!(f, "Raster Names:\n{}", rasters)?;
            }
            if !self.rasters.is_empty() {
                writeln!(f, "Raster Definitions:")?;
                for r in &self.rasters {
                    writeln!(f, "    {}", r)?;
                }
                write!(f, "\n")?;
            }
        }

        Ok(())
    }
}

/// Describes the different triangle types that can be encoded in the DSF.
#[derive(Debug)]
enum TriType {
    /// Three verticies form one triangle.
    Triangle,
    /// N+2 verticies form N triangles (like OpenGL.)
    TriangleStrip,
    /// N+2 verticies form N triangles (like OpenGL.)
    TriangleFan,
}

#[allow(unused_variables)] // So trait func parameters can avoid the leading _ for nop defaults.
trait Callbacks {
    fn accept_terrain_def(&mut self, value: &str) {}
    fn accept_object_def(&mut self, value: &str) {}
    fn accept_polygon_def(&mut self, value: &str) {}
    fn accept_network_def(&mut self, value: &str) {}
    fn accept_raster_def(&mut self, value: &str) {}

    fn accept_property(&mut self, name: &str, value: &str) {}

    /* "These functions build patches. You receive a start patch, then a set of homogenous
     * triangles, then an end patch. All patch vertices must match the number of coordinates passed
     * in coord_depth."
     */
    fn begin_patch(
        &mut self,
        terrain_type: u32,
        near_lod: f64,
        far_lod: f64,
        flags: u8,
        coord_depth: usize,
    ) {
    }
    fn begin_primitive(&mut self, kind: TriType) {}
    fn add_patch_vertex(&mut self, coordinates: &[f64]) {}
    fn end_primitive(&mut self) {}
    fn end_patch(&mut self) {}

    /// "This function adds an object. All objects take two coordinates."
    /// coordinates are 3 or 4 values: Lon Lat Rot [MSL]
    /// idx refers to the current "type" or "definition" -- the index into the OBJECT table.
    fn add_object_with_mode(&mut self, idx: u32, coordinates: &[f64], mode: ElevMode) {}

    /* "This function adds a complete chain. ALl chains take 3 coordinates for non-curved nodes and
     * 6 coordinates for curved nodes."
     */
    // coordinates: "lon lat el, start node ID, shape lon lat el"
    // network_type: an index (current_definition)
    fn begin_segment(&mut self, net_type: u32, subtype: u8, coordinates: &[f64], curved: bool) {}
    fn add_segment_shape_point(&mut self, coords: &[f64], curved: bool) {}
    fn end_segment(&mut self, coords: &[f64], curved: bool) {}

    /* "These functions add polygons. You'll get at least one winding per polygon. All polygons
     * take two coordinates. */
    // poly_type is the same as current_definition / the index into the poly table
    fn begin_polygon(&mut self, poly_type: u32, param: u16, coord_depth: usize) {}
    fn begin_polygon_winding(&mut self) {}
    fn add_polygon_point(&mut self, coordinates: &[f64]) {}
    fn end_polygon_winding(&mut self) {}
    fn end_polygon(&mut self) {}

    fn add_raster_data(&mut self, data: &RasterData) {}

    fn set_filter(&mut self, filter_index: i32) {}
}

struct NopCallbacks {}
impl Callbacks for NopCallbacks {}

/// This object tracks which objects / definitions are referenced by commands.
///
/// This is kept separate from the main callbacks processing / parsing so that DSF::process_cmds
/// can be const.
#[derive(Debug, Default)]
struct ObjectRefCounter {
    object_refs: Vec<usize>,
    terrain_refs: Vec<usize>,
    network_refs: Vec<usize>,
    poly_refs: Vec<usize>,
}

impl ObjectRefCounter {
    fn add(v: &mut Vec<usize>, i: usize, x: usize) {
        if v.len() <= i {
            v.resize(i + 1, 0);
        }
        v[i] += x;
    }
}

impl Callbacks for ObjectRefCounter {
    fn begin_patch(&mut self, terrain_type: u32, _: f64, _: f64, _: u8, _: usize) {
        Self::add(&mut self.terrain_refs, terrain_type as usize, 1);
    }

    fn add_object_with_mode(&mut self, idx: u32, _: &[f64], _: ElevMode) {
        Self::add(&mut self.object_refs, idx as usize, 1);
    }

    fn begin_segment(&mut self, net_type: u32, _: u8, _: &[f64], _: bool) {
        Self::add(&mut self.network_refs, net_type as usize, 1);
    }

    fn begin_polygon(&mut self, poly_type: u32, _: u16, _: usize) {
        Self::add(&mut self.poly_refs, poly_type as usize, 1);
    }
}

/// Produces a text version of the DSF that matches DSFTool's dsf2text mode.
struct DsfToTextConverter {
    /// The file contents are accumulated here.
    out: String,
}

impl Callbacks for DsfToTextConverter {
    fn accept_property(&mut self, name: &str, value: &str) {
        writeln!(self.out, "PROPERTY {} {}", name, value).unwrap();
    }

    fn accept_terrain_def(&mut self, value: &str) {
        writeln!(self.out, "TERRAIN_DEF {}", value).unwrap();
    }

    fn accept_object_def(&mut self, value: &str) {
        writeln!(self.out, "OBJECT_DEF {}", value).unwrap();
    }

    fn accept_polygon_def(&mut self, value: &str) {
        writeln!(self.out, "POLYGON_DEF {}", value).unwrap();
    }

    fn accept_network_def(&mut self, value: &str) {
        writeln!(self.out, "NETWORK_DEF {}", value).unwrap();
    }

    fn accept_raster_def(&mut self, value: &str) {
        writeln!(self.out, "RASTER_DEF {}", value).unwrap();
    }

    fn begin_patch(
        &mut self,
        terrain_type: u32,
        near_lod: f64,
        far_lod: f64,
        flags: u8,
        coord_depth: usize,
    ) {
        // The {:.6} here for LODs is to make the output byte-identical to the reference, which uses
        // %lf. It might create diffs when the LODs aren't integers.
        writeln!(
            self.out,
            "BEGIN_PATCH {} {:.6} {:.6} {} {}",
            terrain_type, near_lod, far_lod, flags, coord_depth
        )
        .unwrap();
    }
    fn begin_primitive(&mut self, kind: TriType) {
        let v = match kind {
            TriType::Triangle => 0,
            TriType::TriangleStrip => 1,
            TriType::TriangleFan => 2,
        };
        writeln!(self.out, "BEGIN_PRIMITIVE {}", v).unwrap();
    }
    fn add_patch_vertex(&mut self, coordinates: &[f64]) {
        write!(self.out, "PATCH_VERTEX").unwrap();
        for c in coordinates {
            write!(self.out, " {:.9}", c).unwrap();
        }
        write!(self.out, "\n").unwrap();
    }
    fn end_primitive(&mut self) {
        writeln!(self.out, "END_PRIMITIVE").unwrap();
    }
    fn end_patch(&mut self) {
        writeln!(self.out, "END_PATCH").unwrap();
    }

    fn add_object_with_mode(&mut self, idx: u32, coordinates: &[f64], mode: ElevMode) {
        let c = coordinates;
        // These use {:.6} where %lf is used in the reference.
        match mode {
            ElevMode::AGL => writeln!(
                self.out,
                "OBJECT_AGL {} {:.9} {:.9} {:.9} {:.6}",
                idx, c[0], c[1], c[3], c[2]
            ),
            ElevMode::MSL => writeln!(
                self.out,
                "OBJECT_MSL {} {:.9} {:.9} {:.9} {:.6}",
                idx, c[0], c[1], c[3], c[2]
            ),
            ElevMode::Draped => writeln!(
                self.out,
                "OBJECT {} {:.9} {:.9} {:.6}",
                idx, c[0], c[1], c[2]
            ),
        }
        .unwrap();
    }

    fn begin_segment(
        &mut self,
        network_type: u32,
        network_subtype: u8,
        coordinates: &[f64],
        curved: bool,
    ) {
        let c = coordinates;
        // The c[3] values below are printed as int -- even though they are double --
        // to match the reference. (Specifically, using {} instead of {:.9}.)
        if !curved {
            writeln!(
                self.out,
                "BEGIN_SEGMENT {} {} {} {:.9} {:.9} {:.9}",
                network_type, network_subtype, c[3], c[0], c[1], c[2]
            )
        } else {
            writeln!(
                self.out,
                "BEGIN_SEGMENT_CURVED {} {} {} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
                network_type, network_subtype, c[3], c[0], c[1], c[2], c[4], c[5], c[6]
            )
        }
        .unwrap();
    }

    fn add_segment_shape_point(&mut self, coords: &[f64], curved: bool) {
        let c = coords;
        if !curved {
            writeln!(self.out, "SHAPE_POINT {:.9} {:.9} {:.9}", c[0], c[1], c[2])
        } else {
            writeln!(
                self.out,
                "SHAPE_POINT_CURVED {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
                c[0], c[1], c[2], c[3], c[4], c[5]
            )
        }
        .unwrap();
    }

    fn end_segment(&mut self, coords: &[f64], curved: bool) {
        let c = coords;
        // The c[3] values below are printed as int -- even though they are double --
        // to match the reference. (Specifically, using {} instead of {:.9}.)
        if !curved {
            writeln!(
                self.out,
                "END_SEGMENT {} {:.9} {:.9} {:.9}",
                c[3], c[0], c[1], c[2]
            )
        } else {
            writeln!(
                self.out,
                "END_SEGMENT_CURVED {} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
                c[3], c[0], c[1], c[2], c[4], c[5], c[6]
            )
        }
        .unwrap();
    }

    /* "These functions add polygons. You'll get at least one winding per polygon. All polygons
     * take two coordinates. */
    // poly_type is the same as current_definition / the index into the poly table
    fn begin_polygon(&mut self, poly_type: u32, param: u16, coord_depth: usize) {
        writeln!(
            self.out,
            "BEGIN_POLYGON {} {} {}",
            poly_type, param, coord_depth
        )
        .unwrap();
    }
    fn begin_polygon_winding(&mut self) {
        writeln!(self.out, "BEGIN_WINDING").unwrap();
    }
    fn add_polygon_point(&mut self, coordinates: &[f64]) {
        write!(self.out, "POLYGON_POINT").unwrap();
        for v in coordinates {
            write!(self.out, " {:.9}", v).unwrap();
        }
        write!(self.out, "\n").unwrap();
    }
    fn end_polygon_winding(&mut self) {
        writeln!(self.out, "END_WINDING").unwrap();
    }
    fn end_polygon(&mut self) {
        writeln!(self.out, "END_POLYGON").unwrap();
    }

    fn add_raster_data(&mut self, data: &RasterData) {
        // The {:.6} for scale and offset are to make the outputs byte-identical to the reference,
        // which uses %lf. It may cause diffs when the values are not integers.
        let path = PathBuf::from("FILENAME"); // FIXME
        writeln!(
            self.out,
            "RASTER_DATA version={} bpp={} flags={} width={} height={} scale={:.6} offset={:.6} {}",
            1,
            data.bytes_per_pixel,
            data.flags,
            data.width,
            data.height,
            data.scale,
            data.offset,
            path.display()
        )
        .unwrap();
    }

    fn set_filter(&mut self, filter_index: i32) {
        let _ = writeln!(self.out, "FILTER {}", filter_index).unwrap();
    }
}

impl DsfInfo {
    // FILE FORMAT NOTES
    // 1. "XPLNEDSF" magic cookie.
    // 2. 32 bit / 4 byte master file format version: little endian int: 0x1.
    // 3. ATOMS: 32-bit atom id; 32-bit unsigned byte count, inc. 8 byte header.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<DsfInfo, Box<dyn Error>> {
        let mut dsf = DsfInfo::default();
        dsf.path = PathBuf::from(path.as_ref());

        // Maybe decompress, check header, and read into vec.
        dsf.read_dsf_bytes(path)?;
        // Check file invariants.
        dsf.verify_checksum()?;
        dsf.verify_file_format()?;

        let root_start: usize = 8 + 4; // Header + file version lengths.
        let root_end = dsf.data.len() - 16; // Ignoring trailing MD5.
        dsf.parse_root_block(root_start, root_end)?;
        Ok(dsf)
    }

    /// Returns whether sim/overlay property is present and set to 1.
    pub fn is_overlay(&self) -> bool {
        return self.properties.as_ref().unwrap().is_overlay();
    }

    /// Returns the sim/creation_agent property, or an empty string.
    pub fn creation_agent(&self) -> String {
        return self
            .properties
            .as_ref()
            .unwrap()
            .get_or_empty("sim/creation_agent");
    }

    /// Returns the number of exclusion entries found in the DSF's properties.
    pub fn num_exclusions(&self) -> usize {
        return self.properties.as_ref().unwrap().num_exclusions();
    }

    /// Returns whether the DSF contains non-empty terrain information.
    pub fn has_terrain(&self) -> bool {
        return !self.terrain_table.is_empty();
    }

    /// Returns the names of the raster layers.
    pub fn raster_names(&self) -> Vec<String> {
        self.raster_definitions.strings.clone()
    }

    /// Returns the given Raster if it exists.
    pub fn raster(&self, name: &str) -> Option<&RasterData> {
        for (i, s) in self.raster_definitions.strings.iter().enumerate() {
            if s == name && i < self.rasters.len() {
                return Some(&self.rasters[i]);
            }
        }
        return None;
    }

    /// Outputs the DSF as a text format used by DSFTool.
    pub fn to_text(&self) -> Result<String, Box<dyn Error>> {
        let mut out = String::with_capacity(10 * 1024 * 1024);
        write!(out, "A\n800\nDSF2TEXT\n\n")?;
        write!(out, "# file: {}\n\n", self.path.display())?;

        let mut converter = DsfToTextConverter {
            out: out, /* moved */
        };

        for (k, v) in self.properties.as_ref().unwrap().pairs() {
            converter.accept_property(k, v);
        }

        for s in &self.terrain_table.strings {
            converter.accept_terrain_def(s);
        }
        for s in &self.object_table.strings {
            converter.accept_object_def(s);
        }
        for s in &self.poly_table.strings {
            converter.accept_polygon_def(s);
        }
        for s in &self.network_table.strings {
            converter.accept_network_def(s);
        }
        for s in &self.raster_definitions.strings {
            converter.accept_raster_def(s);
        }
        // Note: it seems sometimes the RASTER_DATA comes before the commands, and sometimes after?
        for r in &self.rasters {
            converter.add_raster_data(&r);
        }

        // Parse/process the commands.
        self.process_with_callbacks(&mut converter)?;

        // Done with the main content. Take back buffer and add footer.
        out = converter.out; /* moved back */
        writeln!(out, "# Result code: 0")?;
        return Ok(out);
    }

    /// Reads the file at path, expecting a DSF file, decompressing (7z) if necessary.
    ///
    /// Validates the content by looking for the "XPLNEDSF" magic cookie, but does not do any
    /// file format validation after that.
    fn read_dsf_bytes<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn Error>> {
        let mut source = File::open(&path)?;
        let source_size = fs::metadata(&path)?.len() as usize;

        // Read / peek at the header bytes.
        let magic_cookie = b"XPLNEDSF";
        let mut header: Vec<u8> = vec![0; magic_cookie.len()];
        use std::io::Read;
        use std::io::Seek;
        source.read(&mut header)?; // or fail overflow
        source.rewind()?;

        // DSF files may be either raw or 7z compressed.
        if &header[0..2] == b"7z" {
            let files = compress_tools::list_archive_files(&mut source)?;
            source.rewind()?;
            if files.len() != 1 {
                bail!(format!(
                    "There is not exactly one compressed file in {}. Found {:?}!",
                    path.as_ref().display(),
                    &files
                ));
            }

            compress_tools::uncompress_archive_file(&mut source, &mut self.data, &files[0])?;
            debug!(
                "Decompressed {} ({} bytes) as 7z into {} bytes.",
                path.as_ref().display(),
                source_size,
                self.data.len()
            );

            if &self.data[0..magic_cookie.len()] != magic_cookie {
                bail!(format!(
                    "Decompressed data for {} has invalid header: {:?}",
                    &path.as_ref().display(),
                    &self.data[0..magic_cookie.len()]
                ));
            }
            self.compressed = true;
        } else if &header[0..magic_cookie.len()] == magic_cookie {
            self.data.reserve(source_size);
            //let mut target: Vec<u8> = vec![0; source_size];
            let count_read = source.read_to_end(&mut self.data)?;
            debug!(
                "Read uncompressed DSF {} ({} bytes) into memory as {} bytes.",
                path.as_ref().display(),
                &source_size,
                count_read
            );
        } else {
            bail!(
                "File {} invalid header! {:?}",
                path.as_ref().display(),
                &header
            );
        }
        Ok(())
    }

    /// Verifies the last 16 bytes of self.data contain the MD5 of the rest.
    fn verify_checksum(&mut self) -> Result<(), Box<dyn Error>> {
        let data_len = self.data.len() - 16;
        let digest = md5::compute(&self.data[0..data_len]);
        let checksum_valid = &digest as &[u8; 16] == &self.data[data_len..];
        if !checksum_valid {
            bail!(format!(
                "Invalid checksum! computed={:x} trailing_data={:x?}",
                &digest,
                &self.data[data_len..]
            ));
        }
        Ok(())
    }

    /// Verifies the master file format is 1.
    fn verify_file_format(&mut self) -> Result<(), Box<dyn Error>> {
        // Extract master file format.
        let format = Self::get_i32_at(&self.data, 8);
        if format != 1 {
            bail!("Invalid master file format {}! File={:?}", format, self);
        }
        Ok(())
    }

    fn get_i32_at(data: &[u8], offset: usize) -> i32 {
        data[offset] as i32
            + ((data[offset + 1] as i32) << 8)
            + ((data[offset + 2] as i32) << 16)
            + ((data[offset + 3] as i32) << 24)
    }

    fn get_u32_at(data: &[u8], offset: usize) -> u32 {
        data[offset] as u32
            + ((data[offset + 1] as u32) << 8)
            + ((data[offset + 2] as u32) << 16)
            + ((data[offset + 3] as u32) << 24)
    }

    fn process_with_callbacks(&self, callbacks: &mut impl Callbacks) -> Result<(), Box<dyn Error>> {
        let root_start: usize = 8 + 4; // Header + file version lengths.
        let root_end = self.data.len() - 16; // Ignoring trailing MD5.

        let (start, len) = self.get_atom(AtomId::CMDS, root_start, root_end);
        self.process_cmds(start + 8, start + len, callbacks)
    }

    /// Parses the contents of the entire file, starting with the root block.
    ///
    /// The parameters indicate the start index and last index (exclusive) of self.data to parse.
    /// These are indices rather than slices to allow methods to cache/own slices of the real data
    /// without upsetting the borrow checker with borrows-of-temporary-slices and such.
    fn parse_root_block(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        self.require_atoms(
            start,
            end,
            &vec![
                AtomId::HEAD,
                AtomId::DEFN,
                AtomId::GEOD,
                AtomId::CMDS,
                AtomId::DEMS,
            ],
        )?;
        assert!(start <= end);
        assert!(end <= self.data.len());
        let mut pos = start;
        while pos < end {
            let atom_id: i32 = Self::get_i32_at(&self.data, pos);
            let atom_len: usize = Self::get_u32_at(&self.data, pos + 4) as usize;

            match AtomId::from_i32(atom_id) {
                AtomId::HEAD => {
                    self.parse_head_atom(pos + 8, pos + atom_len)?;
                }
                AtomId::DEFN => {
                    self.parse_defn_atom(pos + 8, pos + atom_len)?;
                }
                AtomId::GEOD => {
                    self.parse_geod_atom(pos + 8, pos + atom_len)?;
                }
                AtomId::CMDS => {
                    self.parse_cmds_atom(pos + 8, pos + atom_len)?;
                }
                AtomId::DEMS => {
                    self.parse_dems_atom(pos + 8, pos + atom_len)?;
                }
                _ => bail!(format!(
                    "Found unknown atom id {:?} at root position.",
                    &atom_id
                )),
            }
            pos += atom_len as usize; // Note that atom_len includes the 8 byte atom header.
        }
        Ok(())
    }

    /// Parses the atom at [start, end) and checks that it has exactly the given AtomIds.
    fn require_atoms(
        &self,
        start: usize,
        end: usize,
        ids: &Vec<AtomId>,
    ) -> Result<(), Box<dyn Error>> {
        let mut expected = ids.clone();
        let mut found = vec![];
        for (id, _, _) in self.get_atoms_in_range(start, end) {
            found.push(id);
        }
        if expected.sort() != found.sort() {
            bail!(format!(
                "Wrong atoms found between [{} .. {})! Expected {:?} and found {:?}.",
                start, end, expected, found
            ));
        }
        Ok(())
    }

    /// Parses that atom at [start, end) and checks that it contains only equal pairs of the given
    /// AtomIds.
    fn require_pairs(
        &self,
        start: usize,
        end: usize,
        ids: &Vec<(AtomId, AtomId)>,
    ) -> Result<(), Box<dyn Error>> {
        // FIXME: This is just silly.
        //
        // The idea is to accumulate atom counts here and then check them at the end.
        // The keys are sentinals for acceptable ids.
        let mut counts = std::collections::BTreeMap::<AtomId, usize>::new();
        for (a, b) in ids {
            counts.insert(*a, 0);
            counts.insert(*b, 0);
        }

        for (id, _, _) in self.get_atoms_in_range(start, end) {
            if !counts.contains_key(&id) {
                bail!(format!(
                    "Unexpected atom id {:?} in [{}, {})",
                    id, start, end
                ));
            }
            *counts.get_mut(&id).unwrap() += 1;
        }

        for (a, b) in ids {
            if counts[a] != counts[b] {
                bail!(format!(
                    "Atom count mismatch {:?}={} and {:?}={} in byte range [{}, {})",
                    a, counts[a], b, counts[b], start, end
                ));
            }
        }
        Ok(())
    }

    fn get_atoms_in_range(&self, start: usize, end: usize) -> Vec<(AtomId, usize, usize)> {
        let mut result = vec![];
        let mut pos = start;
        while pos < end {
            let atom_id = AtomId::from_i32(Self::get_i32_at(&self.data, pos));
            let atom_len: usize = Self::get_u32_at(&self.data, pos + 4) as usize;
            result.push((atom_id, pos, atom_len));
            pos += atom_len as usize; // Note that atom_len includes the 8 byte atom header.
        }
        return result;
    }

    fn get_atom(&self, id: AtomId, start: usize, end: usize) -> (usize, usize) {
        for (i, offset, len) in self.get_atoms_in_range(start, end) {
            if i == id {
                return (offset, len);
            }
        }
        panic!(
            "get_atom({:?}) called when there was no atom with that id in [{}..{})",
            id, start, end
        );
    }

    // =============== HEAD and subatoms ================== //

    /// Parses the contents of a HEAD atom from self.data[start..end].
    fn parse_head_atom(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        self.require_atoms(start, end, &vec![AtomId::PROP])?;

        for (id, offset, len) in self.get_atoms_in_range(start, end) {
            match id {
                AtomId::PROP => {
                    let prop_table =
                        StringTable::new(&self.data[offset + 8..offset + len], offset)?;
                    self.properties = Some(PropertiesAtom::new(prop_table)?);
                }
                _ => unreachable!(), // already checked with require_atoms
            }
        }
        Ok(())
    }

    // =============== DEFN and subatoms ================== //

    /// Parses the contents of a DEFN atom from self.data[start..end].
    fn parse_defn_atom(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        self.require_atoms(
            start,
            end,
            &vec![
                AtomId::TERT,
                AtomId::OBJT,
                AtomId::POLY,
                AtomId::NETW,
                AtomId::DEMN,
            ],
        )?;

        for (id, offset, len) in self.get_atoms_in_range(start, end) {
            let slice = &self.data[offset + 8..offset + len];
            match id {
                AtomId::TERT => self.terrain_table = StringTable::new(slice, offset)?,
                AtomId::OBJT => self.object_table = StringTable::new(slice, offset)?,
                AtomId::POLY => self.poly_table = StringTable::new(slice, offset)?,
                AtomId::NETW => self.network_table = StringTable::new(slice, offset)?,
                AtomId::DEMN => self.raster_definitions = StringTable::new(slice, offset)?,
                _ => unreachable!(), // already checked with require_atoms
            }
        }
        Ok(())
    }

    // =============== GEOD and subatoms ================== //
    /// Parses the contents of a GEOD atom from self.data[start..end].
    fn parse_geod_atom(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        self.require_pairs(
            start,
            end,
            &vec![(AtomId::POOL, AtomId::SCAL), (AtomId::PO32, AtomId::SC32)],
        )?;
        trace!("parse_geod_atom(): start={} end={}", start, end);

        // Two-pass decode, with the scales first. This way the points can either be skipped
        // or fully decoded when each pool is encountered.

        for (id, offset, len) in self.get_atoms_in_range(start, end) {
            let mut reader = DataReader::new(&self.data[offset..offset + len]);
            // Consume atom header. TODO: Remove once atom iter uses readers.
            assert_eq!(id as u32, reader.read_u32());
            assert_eq!(len, reader.read_u32() as usize);

            if id != AtomId::SCAL && id != AtomId::SC32 {
                continue;
            }
            let mut pool = Pool::new();
            while !reader.done() {
                // The multiplier/scale and offset are f32 but all math is done as f64.
                let mult = reader.read_f32();
                let offset = reader.read_f32();
                pool.add_plane(mult as f64, offset as f64);
            }
            if id == AtomId::SCAL {
                self.pools.push(pool);
            } else {
                self.po32s.push(pool);
            }
        }

        let mut num_pools_seen = 0;
        let mut num_po32s_seen = 0;
        for (id, offset, len) in self.get_atoms_in_range(start, end) {
            let mut reader = DataReader::new(&self.data[offset..offset + len]);
            // Consume atom header. TODO: Remove once atom iter uses readers.
            assert_eq!(id as u32, reader.read_u32());
            assert_eq!(len, reader.read_u32() as usize);

            if id != AtomId::POOL && id != AtomId::PO32 {
                continue;
            }

            if id == AtomId::POOL {
                // POOL -- elements are uint16
                trace!(
                    "Found POOL {}! data from {} to {} ({} bytes).",
                    num_pools_seen,
                    offset + 8,
                    offset + len,
                    len - 8
                );
                let pool = &mut self.pools[num_pools_seen];
                pool.set_offsets(offset + 8, offset + len);
                pool.decode::<u16>(&mut reader);
                num_pools_seen += 1;
            } else {
                // PO32 -- elements are uint32
                trace!(
                    "Found PO32 {}! data starts at {} with len {} ({} pts)",
                    num_po32s_seen,
                    offset + 8,
                    len - 8,
                    (len - 8) / 4
                );

                let pool = &mut self.po32s[num_po32s_seen];
                pool.set_offsets(offset + 8, offset + len);
                pool.decode::<u32>(&mut reader);
                num_po32s_seen += 1;
            }
        }
        Ok(())
    }

    // =============== DEMS and subatoms ================== //
    fn parse_dems_atom(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        self.require_pairs(start, end, &vec![(AtomId::DEMI, AtomId::DEMD)])?;
        let mut num_demds = 0;

        for (id, ofst, len) in self.get_atoms_in_range(start, end) {
            let mut reader = DataReader::new(&self.data[ofst..ofst + len]);
            // Consume atom header. TODO: Remove once atom iter uses readers.
            assert_eq!(id as u32, reader.read_u32());
            assert_eq!(len, reader.read_u32() as usize);

            match id {
                AtomId::DEMI => {
                    let mut raster = RasterData::default();

                    let version = reader.read_u8();
                    raster.bytes_per_pixel = reader.read_u8();
                    raster.flags = reader.read_u16();
                    raster.width = reader.read_u32();
                    raster.height = reader.read_u32();
                    raster.scale = reader.read_f32();
                    raster.offset = reader.read_f32();
                    assert!(reader.done());
                    if version != 1 {
                        bail!(format!(
                            "Encountered DEMI atom with version {}. Expected 1.",
                            version
                        ));
                    }
                    match (raster.flags & 0x3, raster.bytes_per_pixel) {
                        (0, 4) => (),
                        (1, 1) => (),
                        (1, 2) => (),
                        (1, 4) => (),
                        (2, 1) => (),
                        (2, 2) => (),
                        (2, 4) => (),
                        _ => bail!(format!(
                            "Invalid DEMI flags and bytes per pixel: {}, {}",
                            raster.flags, raster.bytes_per_pixel
                        )),
                    };
                    // TODO: "A flag value of 4 (bit 3) defines the data as post-centric, as
                    // opposed to area-centric. In post-centric data, the pixel values at the edges
                    // of the DEM exactly lie on the geometric boundary of the DSF.  In
                    // point-centric data, the outer edge of the pixel rectangles lie on the
                    // geometric boundary of the DSF.
                    debug!("{}", &raster);
                    self.rasters.push(raster);
                }
                AtomId::DEMD => {
                    debug!("DEMD! Raw data starting at {} for {} bytes.", ofst, len);
                    assert!(num_demds < self.rasters.len());
                    let raster = &mut self.rasters[num_demds];
                    raster.data = (&self.data[ofst + 8..ofst + len]).to_vec();
                    num_demds += 1;
                }
                _ => unreachable!(), // checked with require_pairs above
            }
        }
        Ok(())
    }

    // =============== CMDS and subatoms ================== //
    fn parse_cmds_atom(&mut self, start: usize, end: usize) -> Result<(), Box<dyn Error>> {
        let mut cb: ObjectRefCounter = Default::default();
        self.process_cmds(start, end, &mut cb)?;
        self.object_table.add_all_refs(&cb.object_refs);
        self.terrain_table.add_all_refs(&cb.terrain_refs);
        self.poly_table.add_all_refs(&cb.poly_refs);
        self.network_table.add_all_refs(&cb.network_refs);
        Ok(())
    }

    /// Decodes the CMDS atom, invoking the given callbacks. The CMDS atom must be in [start, end).
    fn process_cmds(
        &self,
        start: usize,
        end: usize,
        callbacks: &mut impl Callbacks,
    ) -> Result<(), Box<dyn Error>> {
        debug!(
            "Processing CMDS atom from [{}..{}) ({} bytes)...",
            start,
            end,
            end - start
        );
        let trace_cmds = false;

        // This is the index into the object/terrain/network/poly table that's usually the first
        // argument, a.k.a. the network_type, etc. Changed by SetDefinition{8,16,32}.
        let mut current_definition = u32::MAX;

        // The selected pools. The network (i.e. road) commands use the 32-bit pools, and the other
        // commands use the 16-bit pools. Changed by PoolSelect.
        let mut current_pool_ptr: Option<&Pool> = None;
        let mut current_po32_ptr: Option<&Pool> = None;
        // The number of planes in the selected pool(s). Cached value of pool.num_planes().
        let mut current_depth: usize = 0;
        let mut current_depth32: usize = 0;

        // Network / road-specific settable parameters.
        let mut road_subtype = u8::MAX;
        let mut junction_offset = u32::MAX;

        // Object-specific settable parameters.
        let mut cur_obj_mode = ElevMode::MSL; // Weirdly, set via Comment command.

        // Terrain-specific settable parameters.
        let mut patch_lod_near: f64 = -1.0;
        let mut patch_lod_far: f64 = -1.0;
        let mut patch_flags: u8 = 0xff;
        let mut patch_open: bool = false; // "open" means a terrain patch cmd was reached.

        let mut reader = DataReader::new(&self.data[start..end]);
        while !reader.done() {
            let command = CommandId::from_u8(reader.read_u8());
            match command {
                CommandId::PoolSelect => {
                    // The coordinate pool select command changes the current coordinate
                    // pool and establishes the longitude and latitude bounds that the first
                    // two fields are interpreted within.
                    let pool = reader.read_u16() as usize;
                    if pool >= self.pools.len() && pool >= self.po32s.len() {
                        bail!(format!("PoolSelect: invalid pool index {}", pool));
                    }
                    if pool < self.pools.len() {
                        current_pool_ptr = Some(&self.pools[pool]);
                        current_depth = self.pools[pool].num_planes();
                    } else {
                        current_pool_ptr = None;
                    }
                    if pool < self.po32s.len() {
                        current_po32_ptr = Some(&self.po32s[pool]);
                        current_depth32 = self.po32s[pool].num_planes();
                    } else {
                        current_po32_ptr = None;
                    }

                    if trace_cmds {
                        trace!("... coordinate pool select idx {}", pool);
                    }
                }
                CommandId::JunctionOffsetSelect => {
                    // The junction offset select command specifies a 32-bit number that
                    // is added to all indices when referencing coordinates for vectors.
                    // This allows the use of a 16-bit vector command for vectors whose
                    // indices are greater than 65535.
                    junction_offset = reader.read_u32();
                    if trace_cmds {
                        trace!("... junction offset select {}", junction_offset);
                    }
                }
                CommandId::SetDefinition8 => {
                    current_definition = reader.read_u8() as u32;
                    if trace_cmds {
                        trace!("... set definition [u8] {}", current_definition);
                    }
                }
                CommandId::SetDefinition16 => {
                    current_definition = reader.read_u16() as u32;
                    if trace_cmds {
                        trace!("... set definition [u16] {}", current_definition);
                    }
                }
                CommandId::SetDefinition32 => {
                    current_definition = reader.read_u32();
                    if trace_cmds {
                        trace!("... set definition [u32] {}", current_definition);
                    }
                }
                CommandId::SetRoadSubtype8 => {
                    // Sets the road subtype for the next vector-segment.
                    road_subtype = reader.read_u8();
                    if trace_cmds {
                        trace!("... set road subtype {}", road_subtype);
                    }
                }
                // These commands place an object on the surface of the mesh. A point
                // pool must be selected and must have at least 3 planes, which are
                // treated as a longitude, latitude, and rotation in degrees.
                CommandId::Object => {
                    let coord_idx = reader.read_u16();
                    let mode = if current_depth == 4 {
                        cur_obj_mode
                    } else {
                        ElevMode::Draped
                    };

                    if trace_cmds {
                        trace!(
                            "... place object {} ({}) at coord idx {} with mode {:?}",
                            current_definition,
                            self.object_table.get(current_definition as usize),
                            coord_idx,
                            &mode
                        );
                    }
                    callbacks.add_object_with_mode(
                        current_definition,
                        current_pool_ptr.as_ref().unwrap().point(coord_idx as usize),
                        mode,
                    );
                }
                CommandId::ObjectRange => {
                    let first_obj = reader.read_u16();
                    let last_obj_plus_1 = reader.read_u16();
                    let mode = if current_depth == 4 {
                        cur_obj_mode
                    } else {
                        ElevMode::Draped
                    };

                    if trace_cmds {
                        trace!(
                            "... place object {} ({}) at coord indices [{}..{}) with mode {:?}",
                            current_definition,
                            self.object_table.get(current_definition as usize),
                            first_obj,
                            last_obj_plus_1,
                            &mode
                        );
                    }
                    for i in first_obj..last_obj_plus_1 {
                        callbacks.add_object_with_mode(
                            current_definition,
                            current_pool_ptr.as_ref().unwrap().point(i as usize),
                            mode,
                        );
                    }
                }
                // The network commands instantiate complete chains and junctions for a
                // network. Networks are formed by instantiating complete chains. The
                // coordinate pool for a network segment must have either four planes
                // (longitude, latitude, elevation, junction ID) or seven planes
                // (adding on longitude, latitude, and a shape point for shaping).
                //
                // Junction IDs are one-based consecutive unique integers. Junction IDs
                // simply indicate what junctions will link to each other; if two junctions
                // have the same ID but different spatial locations (based on the first
                // three planes after transform), this is an error.
                //
                // The junction ID zero indicates a shape point, meaning a coordinate that
                // changes the shape of a vector but is not a junction.
                //
                // All network command except for the network-chain-32 command add the
                // junction offset to all indices.
                CommandId::NetworkChain => {
                    // This command creates one or more complete chains, using all
                    // of the vertices that are specifically enumerated. Complete chains
                    // are started or ended based on the presence of a non-zero junction ID.
                    let count = reader.read_u8(); // number of coords

                    // Note: this uses 32-bit pools.
                    let has_curve = current_depth32 >= 7;

                    if trace_cmds {
                        trace!(
                            "... network chain of {} ({}), subtype {}, with {} indices, \
                                  junction offset {}, and curve {}",
                            current_definition,
                            self.network_table.get(current_definition as usize),
                            road_subtype,
                            count,
                            junction_offset,
                            has_curve
                        );
                    }

                    for i in 0..count {
                        let idx = (reader.read_u16() as u32 + junction_offset) as usize;
                        let coords = current_po32_ptr.as_ref().unwrap().point(idx);
                        if coords[3] != 0.0 {
                            if i > 0 {
                                callbacks.end_segment(coords, has_curve);
                            }
                            if i < (count - 1) {
                                callbacks.begin_segment(
                                    current_definition,
                                    road_subtype,
                                    coords,
                                    has_curve,
                                );
                            }
                        } else {
                            callbacks.add_segment_shape_point(coords, has_curve);
                        }
                    }
                }
                CommandId::NetworkChainRange => {
                    // This command creates one or more complete chains, using all of the
                    // vertices within the range specified. Complete chains are started or
                    // ended based on the presence of a non-zero junction ID.
                    let first_idx = reader.read_u16();
                    let last_idx_p1 = reader.read_u16();

                    let has_curve = current_depth32 >= 7;

                    if trace_cmds {
                        trace!(
                            "... network chains of {} ({}), subtype {}, curved={}, \
                             from offsets [{}..{}) (junction offset {})",
                            current_definition,
                            self.network_table.get(current_definition as usize),
                            road_subtype,
                            has_curve,
                            first_idx as u32 + junction_offset,
                            last_idx_p1 as u32 + junction_offset,
                            junction_offset
                        );
                    }

                    for i in first_idx..last_idx_p1 {
                        let idx = (i as u32 + junction_offset) as usize;
                        let coords = current_po32_ptr.as_ref().unwrap().point(idx);
                        if coords[3] != 0.0 {
                            if i != first_idx {
                                callbacks.end_segment(coords, has_curve);
                            }
                            if i != last_idx_p1 - 1 {
                                callbacks.begin_segment(
                                    current_definition,
                                    road_subtype,
                                    coords,
                                    has_curve,
                                );
                            }
                        } else {
                            callbacks.add_segment_shape_point(coords, has_curve);
                        }
                    }
                }
                CommandId::NetworkChain32 => {
                    // This command creates one or more complete chains, but rather than using
                    // 16-bit indices and the junction offset, they use explicit 32-bit indices and
                    // no offset. Use this command to create a vector when the indices span a range
                    // of more than 65536.
                    let count = reader.read_u8(); // number of coords

                    let has_curve = current_depth32 >= 7;

                    if trace_cmds {
                        trace!(
                            "... network chain of {} ({}), subtype {}, with {} exact indices \
                                  and curve {}",
                            current_definition,
                            self.network_table.get(current_definition as usize),
                            road_subtype,
                            count,
                            has_curve
                        );
                    }
                    for i in 0..count {
                        let idx = reader.read_u32() as usize;
                        let coords = current_po32_ptr.as_ref().unwrap().point(idx);
                        if coords[3] != 0.0 {
                            if i > 0 {
                                callbacks.end_segment(coords, has_curve);
                            }
                            if i < (count - 1) {
                                callbacks.begin_segment(
                                    current_definition,
                                    road_subtype,
                                    coords,
                                    has_curve,
                                );
                            }
                        } else {
                            callbacks.add_segment_shape_point(coords, has_curve);
                        }
                    }
                }
                // The polygon commands instantiate polygon primitives on the surface of the mesh.
                // The selected plane must have at least two planes, which are interpreted as
                // longitude and latitude. A per-polygon 16-bit parameter is interpreted based on
                // the polygon definition.
                CommandId::Polygon => {
                    // Instantiates a single polygon.
                    let parameter = reader.read_u16();
                    let count = reader.read_u8();
                    if trace_cmds {
                        trace!(
                            "... polygon of {} ({}) and param={} with {} indices",
                            current_definition,
                            self.poly_table.get(current_definition as usize),
                            parameter,
                            count
                        );
                    }

                    callbacks.begin_polygon(current_definition, parameter, current_depth);
                    callbacks.begin_polygon_winding();
                    for _ in 0..count {
                        let idx = reader.read_u16();
                        callbacks.add_polygon_point(
                            current_pool_ptr.as_ref().unwrap().point(idx as usize),
                        );
                    }
                    callbacks.end_polygon_winding();
                    callbacks.end_polygon();
                }
                CommandId::PolygonRange => {
                    // Instantiates a polygon through a contiguous range of vertices.
                    let parameter = reader.read_u16();
                    let first_idx = reader.read_u16();
                    let last_idx_p1 = reader.read_u16();
                    if trace_cmds {
                        trace!(
                            "... polygon of {} ({}) and param={} range from {} through {}",
                            current_definition,
                            self.poly_table.get(current_definition as usize),
                            parameter,
                            first_idx,
                            last_idx_p1
                        );
                    }
                    callbacks.begin_polygon(current_definition, parameter, current_depth);
                    callbacks.begin_polygon_winding();
                    for i in first_idx..last_idx_p1 {
                        callbacks.add_polygon_point(
                            current_pool_ptr.as_ref().unwrap().point(i as usize),
                        );
                    }
                    callbacks.end_polygon_winding();
                    callbacks.end_polygon();
                }
                CommandId::NestedPolygon => {
                    // Instantiates a series of polygons, each with a distinct winding.
                    let parameter = reader.read_u16();
                    let num_windings = reader.read_u8();

                    if trace_cmds {
                        trace!(
                            "... nested polygon of {} ({}) and param={} with {} windings",
                            current_definition,
                            self.poly_table.get(current_definition as usize),
                            parameter,
                            num_windings
                        );
                    }

                    callbacks.begin_polygon(current_definition, parameter, current_depth);
                    for _i in 0..num_windings {
                        callbacks.begin_polygon_winding();
                        let count = reader.read_u8();
                        for _ in 0..count {
                            let idx = reader.read_u16();
                            callbacks.add_polygon_point(
                                current_pool_ptr.as_ref().unwrap().point(idx as usize),
                            );
                        }
                        callbacks.end_polygon_winding();
                    }
                    callbacks.end_polygon();
                }
                CommandId::NestedPolygonRange => {
                    // This command instantiates a series of polygons with distinct windings
                    // using a list of contiguous ranges. Each index starts a winding except
                    // for the last, which is one past the end of the polygon’s last point.
                    let parameter = reader.read_u16();
                    let count = reader.read_u8();

                    if trace_cmds {
                        trace!(
                            "... nested polygon range of {} ({}) and param={} with {} vertices",
                            current_definition,
                            self.poly_table.get(current_definition as usize),
                            parameter,
                            count
                        );
                    }

                    let mut index1 = reader.read_u16();
                    callbacks.begin_polygon(current_definition, parameter, current_depth);
                    for _ in 0..count {
                        callbacks.begin_polygon_winding();
                        let index2 = reader.read_u16();
                        for i in index1..index2 {
                            callbacks.add_polygon_point(
                                current_pool_ptr.as_ref().unwrap().point(i as usize),
                            );
                        }
                        callbacks.end_polygon_winding();
                        index1 = index2;
                    }
                    callbacks.end_polygon();
                }
                // MESH COMMANDS
                //
                // The mesh commands instantiate the terrain mesh as triangles. The mesh commands
                // take planar data with at least 5 parameters, corresponding to to longitude,
                // latitude, elevation, and a normal. Additional parameters are used to texture the
                // patch based on the .ter terrain type definition file.
                //
                // ... lots of stuff ...
                //
                CommandId::TerrainPatch
                | CommandId::TerrainPatchFlags
                | CommandId::TerrainPatchFlagsLOD => {
                    // These commands indicate that a new terrain patch is being created. They vary
                    // on whether some param values are kept or changed. The patch remains "open"
                    // until the next TerrainPatch command or the end of the file.
                    if command != CommandId::TerrainPatch {
                        // PatchFlags and PatchFlagsLOD.
                        // The flags are:
                        //   1: Physical: If set, this patch is used for collision detection.
                        //   2: Overlay: If set, this patch is drawn over another patch.
                        patch_flags = reader.read_u8();
                    }
                    if command == CommandId::TerrainPatchFlagsLOD {
                        // A new LOD range in meters.
                        patch_lod_near = reader.read_f32() as f64;
                        patch_lod_far = reader.read_f32() as f64;
                    }

                    if trace_cmds {
                        trace!("... new terrain patch of {} ({}) with near {} far {} flags {} and depth {}",
                            current_definition,
                            self.terrain_table.get(current_definition as usize),
                            patch_lod_near, patch_lod_far, patch_flags,
                            current_pool_ptr.unwrap().num_planes(),
                            );
                    }
                    if patch_open {
                        callbacks.end_patch();
                    }
                    callbacks.begin_patch(
                        current_definition,
                        patch_lod_near,
                        patch_lod_far,
                        patch_flags,
                        current_depth,
                    );
                    patch_open = true;
                }
                CommandId::Triangle => {
                    // This command creates one or more specific triangle for a terrain patch.
                    // Triangles must have clockwise rotation as seen from above for all triangle
                    // primitives.
                    let coord_count = reader.read_u8();

                    if trace_cmds {
                        trace!("... patch triangles. num coords {}", coord_count);
                    }
                    callbacks.begin_primitive(TriType::Triangle);
                    for _ in 0..coord_count {
                        let idx = reader.read_u16();
                        callbacks.add_patch_vertex(
                            current_pool_ptr.as_ref().unwrap().point(idx as usize),
                        );
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleCrossPool => {
                    // This command creates one triangle from multiple terrain pools. This is the
                    // same as the command above, except that a pool index is provided per vertex.
                    let coord_count = reader.read_u8();

                    if trace_cmds {
                        trace!("... cross-pool triangle patch. num coords {}", coord_count);
                    }
                    callbacks.begin_primitive(TriType::Triangle);
                    for _ in 0..coord_count {
                        let pool_id = reader.read_u16() as usize;
                        let idx = reader.read_u16() as usize;
                        if pool_id >= self.pools.len() {
                            bail!("Invalid pool index for TriangleCrossPool command.");
                        }
                        let pool = &self.pools[pool_id];
                        callbacks.add_patch_vertex(pool.point(idx));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleRange => {
                    // This command creates a number of triangles based on the inclusive range
                    // of coordinate indices. THe range must be a multiple of 3. Each set of 3
                    // adjacent vertices is treated as a triangle.
                    let first_index = reader.read_u16();
                    let last_index_p1 = reader.read_u16();

                    if trace_cmds {
                        trace!(
                            "... patch triangle range [{} .. {})",
                            first_index,
                            last_index_p1
                        );
                    }
                    callbacks.begin_primitive(TriType::Triangle);
                    for i in first_index..last_index_p1 {
                        callbacks
                            .add_patch_vertex(current_pool_ptr.as_ref().unwrap().point(i as usize));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleStrip => {
                    // This command creates a triangle strip. A triangle strip is a series of
                    // adjacent triangles that share two common vertices; for a series of points
                    // 1,2,3,4,5 as a triangle strip is equivalent to the trianges 123, 234, 345...
                    let count = reader.read_u8();

                    if trace_cmds {
                        trace!("... patch triangle strip {}", count);
                    }
                    callbacks.begin_primitive(TriType::TriangleStrip);
                    for _ in 0..count {
                        let idx = reader.read_u16() as usize;
                        callbacks.add_patch_vertex(current_pool_ptr.as_ref().unwrap().point(idx));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleStripCrossPool => {
                    // This command creates a triangle strip, except the point pool is specified
                    // per verted rather than referencing the current coordinate pool.
                    let count = reader.read_u8();

                    if trace_cmds {
                        trace!("... patch triangle strip cross-pool {}", count);
                    }
                    callbacks.begin_primitive(TriType::TriangleStrip);
                    for _ in 0..count {
                        let pool_id = reader.read_u16() as usize;
                        let idx = reader.read_u16() as usize;
                        if pool_id >= self.pools.len() {
                            bail!("Invalid pool index for TriangleCrossPool command.");
                        }
                        let pool = &self.pools[pool_id];
                        callbacks.add_patch_vertex(pool.point(idx));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleStripRange => {
                    // Creates a triangle strip for a series of consecutive coordinates.
                    let first_index = reader.read_u16();
                    let last_index_p1 = reader.read_u16();

                    if trace_cmds {
                        trace!(
                            "... patch triangle strip range [{} .. {})",
                            first_index,
                            last_index_p1
                        );
                    }
                    callbacks.begin_primitive(TriType::TriangleStrip);
                    for i in first_index..last_index_p1 {
                        callbacks
                            .add_patch_vertex(current_pool_ptr.as_ref().unwrap().point(i as usize));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleFan => {
                    // This command creates a triangle fan. A triangle fan is a series of adjacent
                    // triangles that share two common vertices; for a series of points 1,2,3,4,5
                    // as a triangle fan is equivalent to the triangles 123, 134, 145, ...
                    let count = reader.read_u8();

                    if trace_cmds {
                        trace!("... patch triangle fan {}", count);
                    }
                    callbacks.begin_primitive(TriType::TriangleFan);
                    for _ in 0..count {
                        let idx = reader.read_u16() as usize;
                        callbacks.add_patch_vertex(current_pool_ptr.as_ref().unwrap().point(idx));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleFanCrossPool => {
                    // This command creates a triangle fan, except the point pool is specified per
                    // vertex rather than referencing the current coordinate pool.
                    let count = reader.read_u8();

                    if trace_cmds {
                        trace!("... patch triangle fan cross-pool {}", count);
                    }
                    callbacks.begin_primitive(TriType::TriangleFan);
                    for _ in 0..count {
                        let pool_id = reader.read_u16() as usize;
                        let idx = reader.read_u16() as usize;
                        if pool_id >= self.pools.len() {
                            bail!("Invalid pool index for TriangleCrossPool command.");
                        }
                        let pool = &self.pools[pool_id];
                        callbacks.add_patch_vertex(pool.point(idx));
                    }
                    callbacks.end_primitive();
                }
                CommandId::TriangleFanRange => {
                    // This command creates a triangle fan for a series of consecutive coordinates.
                    let first_index = reader.read_u16();
                    let last_index_p1 = reader.read_u16();

                    if trace_cmds {
                        trace!(
                            "... patch triangle fan range [{} .. {})",
                            first_index,
                            last_index_p1
                        );
                    }
                    callbacks.begin_primitive(TriType::TriangleFan);
                    for i in first_index..last_index_p1 {
                        callbacks
                            .add_patch_vertex(current_pool_ptr.as_ref().unwrap().point(i as usize));
                    }
                    callbacks.end_primitive();
                }
                // Comment Commands. These commands allow arbitrary data to be embedded in a DSF
                // file. The length field tells the length of the following comment data not
                // including the length field itself.
                //
                // Special formats: If the comment starts with a 0x1 or 0x2 unsigned short, then it
                // indicates a filter or elevation preference.
                CommandId::Comment8 | CommandId::Comment16 | CommandId::Comment32 => {
                    let comment_len = match command {
                        CommandId::Comment8 => reader.read_u8() as u32,
                        CommandId::Comment16 => reader.read_u16() as u32,
                        CommandId::Comment32 => reader.read_u32(),
                        _ => unreachable!(),
                    };
                    const DSF_COMMENT_FILTER: u16 = 1;
                    const DSF_COMMENT_AGL: u16 = 2;
                    let mut bytes_remaining = comment_len as usize;
                    if comment_len >= 2 {
                        let comment_type = reader.read_u16();
                        bytes_remaining -= 2;
                        if comment_type == DSF_COMMENT_FILTER && bytes_remaining == 4 {
                            let filter_idx = reader.read_i32();
                            bytes_remaining -= 4;
                            if trace_cmds {
                                trace!("... comment DSF_COMMENT_FILTER with filter {}", filter_idx);
                            }
                            callbacks.set_filter(filter_idx);
                        }
                        if comment_type == DSF_COMMENT_AGL && bytes_remaining == 4 {
                            let want_agl = reader.read_i32() != 0;
                            bytes_remaining -= 4;

                            if want_agl {
                                cur_obj_mode = ElevMode::AGL;
                            } else {
                                cur_obj_mode = ElevMode::MSL;
                            }
                            if trace_cmds {
                                trace!(
                                    "... comment DSF_COMMENT_AGL changed mode to {:?}",
                                    cur_obj_mode
                                );
                            }
                        }
                    }

                    // N * u8 comment data
                    reader.skip_n_bytes(bytes_remaining);
                    if trace_cmds {
                        trace!("... comment of {} bytes", comment_len);
                    }
                }
                CommandId::Reserved => {
                    bail!("!!! Unexpected reserved command encountered while parsing.");
                }
            }
        }
        if patch_open {
            callbacks.end_patch();
        }
        Ok(())
    }
}
