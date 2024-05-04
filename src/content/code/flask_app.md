---
title: 'Building a tiny machine learning API'
description: 'Creating inference enpoints with flask and huggingface'
pubDate: 'Apr 22 2024'
tags: ["python"]
---
I've been using the <a href="https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/">NMS algorithm</a> for some time now as a post-processing step in detection models without ever thinking about it. I like understanding all the tools I use, so I throught it would be fun to try to implement the algo by hand to make sure I fully grasped the idea. Around this time I also happened to be learning Rust so I figured I could kill two birds with one stone.

The parts I'm going to tackle in this post are:
* artificial data generation 
* IoU computation 
* NMS implementation

## Code 

### Generating the data
We first need some boxes and scores. We could do this by using some popular detection datasets, but its easier instead to randomly generate the data ourselves.
```rust 
//lib.rs 
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

pub mod utils {
    use super::*;
    use rand::distributions::Uniform;
    use rand::{self, Rng};
    use std::fs;
    use std::io::{BufWriter, Write};

    pub fn read_boxes_and_scores(dir: &str) -> Result<(Vec<BBox>, Vec<f32>), Box<dyn std::error::Error>> {
        let boxes_ = fs::read_to_string(format!("{}/boxes.txt", dir))?;
        let boxes: Vec<BBox> = serde_json::from_str(&boxes_)?;

        let scores_ = fs::read_to_string(format!("{}/scores.txt", dir))?;
        let scores: Vec<f32> = serde_json::from_str(&scores_)?;

        Ok((boxes, scores))
    }

    pub fn generate_boxes_and_scores(
        nboxes: usize,
        dir: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut boxes = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut gen = rand::thread_rng();
        let dist = Uniform::new(0.0, 1.0);

        while boxes.len() < nboxes {
            let box_vec: Vec<f32> = (&mut gen).sample_iter(dist).take(4).collect();

            if box_vec[0] < box_vec[2] && box_vec[1] < box_vec[3] {
                let bbox = BBox::from(box_vec);
                if bbox.area() > 0.1 {
                    boxes.push(bbox);

                    scores.push((&mut gen).sample(dist));
                }
            }
        }

        //save
        let mut file = fs::File::create(format!("{}/boxes.txt", dir))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &boxes);

        let mut file = fs::File::create(format!("{}/scores.txt", dir))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &scores);

        Ok(())
    }
}
```

I chose to not include the box score as a field in the `BBox` struct to keep the interface similar to <a href="https://pytorch.org/vision/main/generated/torchvision.ops.nms.html">PyTorch's</a> which is important for consistency when we compare the two later as certain sorting ops add overhead that should be consistent across tests.

To serialize and deserialize the resulting vector of `BBox` objects we add `serde` and `serde_json` to our dependencies. From what I learned, <a href = "https://serde.rs/">Serde</a> is the defacto crate for dealing with json and other structured data serialization in Rust. 

I also learned while doing this that the `BufWriter` lets you avoid creating a copy of your data in-memory as opposed to doing something like `serde_json::to_string(boxes)?` and a file write on the JSON string. In our case the data we're dealing with isn't big enough to worry about this, but its still nice to know. 

### IoU calculation 
Let's first start by defining a few methods for the `BBox` struct to compute basic properties like area or to create new `BBox` instances. 

```rust 
//lib.rs
impl BBox {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox { x1, y1, x2, y2 }
    }
}
impl From<Vec<f32>> for BBox {
    fn from(nums: Vec<f32>) -> BBox {
        assert!(nums.len() >= 4);
        BBox::new(nums[0], nums[1], nums[2], nums[3])
    }
}
```


With the bounding box methods in place we can introduce the IoU operator which computes a relative degree of overlap between two different boxes. The calculation comes from basic set theory and just divides the area of the intersection by the total area enclosed in the union of both boxes (hence the name).

```rust 
//lib.rs 
pub fn iou(box1: &BBox, box2: &BBox) -> f32 {
    let intersection = BBox::new(
        box1.x1.max(box2.x1),
        box1.y1.max(box2.y1),
        box1.x2.min(box2.x2),
        box1.y2.min(box2.y2),
    );
    if intersection.x2 <= intersection.x1 || intersection.y1 >= intersection.y2 {
        return 0.0;
    }
    let union_area: f32 = box1.area() + box2.area() - intersection.area();
    intersection.area() / union_area
}
```

### NMS
Okay! Time for the actual NMS algorithm. The main trick here is to first order the boxes by their confidence scores before doing the filtering. This way you don't have to keep a running variable tracking the max seen so far. 

```rust 
//lib.rs 
pub fn nms(boxes: &mut Vec<BBox>, scores: &Vec<f32>, iou_threshold: f32) -> Vec<BBox> {
    assert_eq!(boxes.len(), scores.len());

    //sort boxes and scores in descending order
    let mut indices = (0..scores.len()).collect::<Vec<usize>>();
    indices.sort_by(|&a, &b| (&scores[a]).partial_cmp(&scores[b]).unwrap());

    let mut sorted_boxes = Vec::new();
    indices.iter().for_each(|&i| sorted_boxes.push(boxes[i]));

    let mut nms_boxes: Vec<BBox> = Vec::new();
    while sorted_boxes.len() > 0 {
        if let Some(some_box) = sorted_boxes.pop() {
            nms_boxes.push(some_box);
            // remove all other boxes with iou above iou_threshold
            sorted_boxes = sorted_boxes
                .into_iter()
                .filter(|x| iou(&x, &some_box) < iou_threshold)
                .collect();
        } else {
            break;
        }
    }
    nms_boxes
}
```
## Results  
Let's actually run the code and time it with the `main.rs` as shown below:

```rust 
//main.rs 
pub mod nms;
use nms::*;

fn main() {
    let _ = utils::generate_boxes_and_scores(10000, "../");
    let mut boxes_and_scores = utils::read_boxes_and_scores("../").unwrap();

    let now = std::time::Instant::now();
    let nms_res = nms(&mut boxes_and_scores.0, &boxes_and_scores.1, 0.5);
    let elapsed = now.elapsed();

    println!(
        "filtered down to {} boxes in {} milliseconds",
        nms_res.len(),
        elapsed.as_millis() 
    );
}
```

```markdown
>filtered down to 130 boxes in 14.49977 milliseconds
```

We can compare the performance of our code to PyTorch's native nms implementation as a quick sanity check. 
```python 
import json 
import time
import torch 
from torchvision.ops import nms 

with open("scores.txt",'r') as f:
  scores = json.loads(f.read())

with open("boxes.txt",'r') as f:
  boxes = json.loads(f.read())

# our boxes were saved as objects from the rust code above ...
boxes = list(map(lambda box: [box['x1'], box['y1'], box['x2'], box['y2']]))

# time it 
start = time.perf_counter()
nms_res = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold = 0.5)
stop = time.perf_counter()

print(f"filtered down to {len(nms_res)} in {1e3*(stop-start)} milliseconds")
```

```markdown 
>filtered down to 130 boxes in 15.97233 milliseconds
```


For thorough benchmarking we should measure multiple times and average, but for a quick check its reassuring to see that the implementation in Rust is on the same magnitude in terms of compute time and that we get the same answer. 
