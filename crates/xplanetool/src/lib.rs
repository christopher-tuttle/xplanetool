use std::error::Error;
use std::path::{Path, PathBuf};

pub fn exp_pngs<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
    let info = dsf::DsfInfo::new(path.as_ref())?;
    for rname in info.raster_names() {
        if let Some(raster) = info.raster(&rname) {
            let path = format!("./{}-new.png", &rname);
            raster_to_png(&raster, &path)?;
            let path = format!("./{}.gif", &rname);
            raster_to_gif(&raster, &path)?;
        }
    }
    Ok(())
}

fn raster_to_png<P: AsRef<Path>>(raster: &dsf::RasterData, path: P) -> Result<(), Box<dyn Error>> {
    let width = raster.width();
    let height = raster.height();
    let mut img = image::ImageBuffer::new(width, height);

    let (min, max) = raster.bounds_unscaled();

    for x in 0..width {
        for y in 0..height {
            let v = raster.unscaled(x, y);

            if v < 0.0 {
                let frac = v / min;
                let stripe = frac * 256.0 - (((frac * 256.0) as i32) as f32);
                let red: u8 = 255 - ((frac * 255.0) as u8);
                let blue: u8 = (stripe * 256.0) as u8;
                img[(x, height - 1 - y)] = image::Rgb([red, 0, blue]);
            } else {
                let frac = v / max;
                let stripe = frac * 256.0 - (((frac * 256.0) as i32) as f32);
                let green: u8 = (frac * 255.0) as u8;
                let blue: u8 = (stripe * 256.0) as u8;
                img[(x, height - 1 - y)] = image::Rgb([0, green, blue]);
            }
        }
    }
    img.save(path.as_ref())?;
    Ok(())
}

struct Rainbow {
    min: f32,
    max: f32,
}

impl Rainbow {
    fn style(&self, v: f32) -> plotters::prelude::ShapeStyle {
        (&plotters::prelude::HSLColor(((v - self.min) / (self.max - self.min)).into(), 1.0, 0.7))
            .into()
    }
}

struct ColorScheme {
    scheme: Vec<(f32, plotters::prelude::RGBColor)>,
}

impl ColorScheme {
    fn ryb() -> ColorScheme {
        use plotters::prelude::*;
        ColorScheme {
            scheme: vec![
                (-4000.0, RGBColor(69, 117, 180)),
                (-3000.0, RGBColor(116, 173, 209)),
                (-2000.0, RGBColor(171, 217, 233)),
                (-1000.0, RGBColor(224, 243, 248)),
                (0.0, RGBColor(255, 255, 191)),
                (1000.0, RGBColor(254, 224, 144)),
                (2000.0, RGBColor(253, 174, 97)),
                (3000.0, RGBColor(244, 109, 67)),
                (4000.0, RGBColor(215, 48, 39)),
            ],
        }
    }

    fn earth() -> ColorScheme {
        use plotters::prelude::*;
        ColorScheme {
            scheme: vec![
                (-4000.0, RGBColor(69, 117, 180)),
                (-3000.0, RGBColor(116, 173, 209)),
                (-2000.0, RGBColor(171, 217, 233)),
                (-1000.0, RGBColor(224, 243, 248)),
                (-10.0, RGBColor(255, 255, 191)),
                (10.0, RGBColor(217, 239, 139)),
                (11.0, RGBColor(26, 152, 80)),
                (2000.0, RGBColor(255, 255, 191)),
                (4000.0, RGBColor(140, 81, 10)),
            ],
        }
    }

    fn style(&self, v: f32) -> plotters::prelude::ShapeStyle {
        use plotters::prelude::*;

        if v < self.scheme[0].0 {
            return self.scheme[0].1.into();
        }

        for i in 1..self.scheme.len() {
            if v >= self.scheme[i].0 {
                continue;
            }
            let base = Self::lerp(
                v,
                self.scheme[i - 1].0,
                self.scheme[i].0,
                &self.scheme[i - 1].1,
                &self.scheme[i].1,
            );
            if v > 0.0 {
                let delta = v as i32 % 100;
                if delta < 10 {
                    return Self::lerp(delta as f32, 0.0, 10.0, &RGBColor(118, 42, 131), &base)
                        .into();
                } else if delta > 90 {
                    return Self::lerp(delta as f32, 90.0, 100.0, &base, &RGBColor(118, 42, 131))
                        .into();
                }
            }
            return base.into();
        }
        return self.scheme[self.scheme.len() - 1].1.into();
    }

    fn lerp(
        v: f32,
        low: f32,
        high: f32,
        low_color: &plotters::prelude::RGBColor,
        high_color: &plotters::prelude::RGBColor,
    ) -> plotters::prelude::RGBColor {
        let frac = (v - low) / (high - low);
        return plotters::prelude::RGBColor(
            (low_color.0 as f32 * (1.0 - frac) + high_color.0 as f32 * frac) as u8,
            (low_color.1 as f32 * (1.0 - frac) + high_color.1 as f32 * frac) as u8,
            (low_color.2 as f32 * (1.0 - frac) + high_color.2 as f32 * frac) as u8,
        );
    }

    fn style_three(&self, v: f32) -> plotters::prelude::ShapeStyle {
        use plotters::prelude::*;
        let max = 4000.0;
        let min = -4000.0;
        let center = 0.0;
        let max_color = RGBColor(140, 81, 10);
        let min_color = RGBColor(1, 102, 94);
        let center_color = RGBColor(245, 245, 245);

        if v < min {
            return min_color.into();
        }
        if v > max {
            return max_color.into();
        }
        fn lerp(
            v: f32,
            low: f32,
            high: f32,
            low_color: &RGBColor,
            high_color: &RGBColor,
        ) -> RGBColor {
            let frac = (v - low) / (high - low);
            return RGBColor(
                (low_color.0 as f32 * (1.0 - frac) + high_color.0 as f32 * frac) as u8,
                (low_color.1 as f32 * (1.0 - frac) + high_color.1 as f32 * frac) as u8,
                (low_color.2 as f32 * (1.0 - frac) + high_color.2 as f32 * frac) as u8,
            );
        }

        if v < center {
            return lerp(v, min, center, &min_color, &center_color).into();
        } else {
            return lerp(v, center, max, &center_color, &max_color).into();
        }
    }
}

fn raster_to_gif<P: AsRef<Path>>(raster: &dsf::RasterData, path: P) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;

    let width = raster.width();
    let height = raster.height();
    let (min, max) = raster.bounds_unscaled();

    // let mut img = image::ImageBuffer::new(width, height);
    let root = BitMapBackend::gif(path.as_ref(), (2000, 1200), 100)?.into_drawing_area();

    let title = path.as_ref().file_name().unwrap().to_string_lossy();
    for step in 0..15 {
        let pitch = step * 10;
        println!("working on pitch {} ...", pitch);
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(&title, ("sans-serif", 20))
            .build_cartesian_3d(0..width, min..max, 0..height)?;
        chart.with_projection(|mut p| {
            p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .max_light_lines(3)
            .draw()?;

        let r = Rainbow { min: min, max: max };
        let s = ColorScheme::earth();

        chart.draw_series(
            SurfaceSeries::xoz(0..width, 0..height, |x, y| raster.unscaled(x, y))
                .style_func(&|&v| s.style(v)), /*
                                               .style_func(&|&v| {
                                                   (&HSLColor((((v - min) / (max - min))).into(), 1.0, 0.7)).into()
                                               }) */
        )?;
        root.present()?;
    }
    Ok(())
}
