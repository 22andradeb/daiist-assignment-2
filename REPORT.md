# Assignment 2 Report

- **Name**: Bernarda Andrade
- **Student ID**: 19119
- **Email**: bandrade.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Business problem, and the cost of being wrong

My family is a small avocado producer in Ecuador. Before fruit is exported,
someone has to look at each avocado and decide whether it is healthy enough to
export, should be rejected, or needs a closer look. This model supports that
decision at the sorting table: it flags avocados showing disease (anthracnose,
Cercospora spot, sunblotch or stem-end rot) so they can be pulled before
packing.

The two mistakes do not cost the same. Rejecting a healthy avocado (false
positive) loses the difference between its export price and what it sells for
on the local market, about €0.30. Exporting a diseased one (false negative)
can ruin a whole batch, because rot spreads to the fruit around it in the box
and the buyer can discount or reject the shipment. I put that at about €20.
A missed diseased avocado is therefore roughly 65 times more costly than a
wrongly rejected healthy one. This is why I care more about recall on the
diseased class than about precision, and why the threshold ends up low.

## Dataset

**Roboflow Universe:** workspace `yaiinternsproject`, project `avocado-foo4c`,
version `1`:
https://universe.roboflow.com/yaiinternsproject/avocado-foo4c/dataset/1
(CC BY 4.0, object detection, YOLOv11 format, images resized to 640×640).

One image is usually a photo of a single avocado (median of one labelled object
per image), with a bounding box around each fruit labelled with its condition.
There are five classes: **Anthracnose**, **Cercospora Spot**, **Healthy**,
**Stem End Rot** and **Sunblotch**.

For the decision I group them: the **positive class is "diseased"** (any of the
four diseases → reject) and the **negative class is "Healthy"** (export). The
model still learns all five classes so I can report each one separately.

This dataset fits the problem because it labels the specific skin diseases a
sorter looks for before export, not just "good" and "bad" fruit, and it
includes a Healthy class (103 test boxes) to measure false alarms against. It
falls short in two ways. First, the photos are not from a sorting table: in
the images I looked at there were studio shots on a white background,
supermarket shelf photos and even collage figures taken from research papers.
Second, one disease, Stem End Rot, has almost no examples (see below).

## What the dataset's composition implies

Computed from the label files (`artifacts/dataset_stats.csv`):

| Split | Images | Anthracnose | Cercospora Spot | Healthy | Stem End Rot | Sunblotch |
|---|---|---|---|---|---|---|
| Train | 6,081 (≈2,027 originals × 3 augmented copies) | 3,060 | 2,397 | 2,574 | **21** (≈7 originals) | 1,080 |
| Valid | 541 | 299 | 216 | 220 | **0** | 58 |
| Test | 268 | 133 | 114 | 103 | **0** | **26** (in 12 images) |
| Median box size, % of image (test) | | 17% | 36% | 34% | n/a | 10% |

Of the 268 test images, 198 contain at least one diseased avocado and 70 are
healthy only. Most images show one avocado that fills a large part of the frame.

**Stem End Rot cannot be evaluated.** It has about 7 original training photos
and none in validation or test, so I have no measured performance for it at
all: I do not know whether the model catches it, only that I cannot say. With
so few examples it has probably learned very little, and as far as I understand
stem-end rot starts at the stem and can be hard to see from outside. Because a
missed diseased avocado is my expensive error (€20), this is a real blind spot.
Everything in this report covers only the four classes in the test set, and
before using this on the farm I would collect and label photos of stem-end rot.

**Sunblotch is too thin to trust.** 26 boxes in 12 images means one or two
mistakes move its precision and recall by several points. The per-class numbers
I can trust are Anthracnose, Cercospora Spot and Healthy, each with 100+ test
boxes.

**The 74% diseased test split does not break the cost analysis.** On my farm I
assume 4% of avocados are diseased. The rates I use, the miss rate (share of
diseased avocados the model passes) and the false-alarm rate (share of healthy
avocados it rejects), are measured within each class, so they do not depend on
the mix. I measure them on the test split and then reweight to 4%:
expected cost per avocado = 4% × miss rate × €20 + 96% × false-alarm rate × €0.30.
What the mix does change is anything that blends the classes. At my chosen
threshold, precision on the test split is 98% (193 of 197 rejections are really
diseased), but at a 4% base rate only about **42% of rejections would be truly
diseased**, so most rejected fruit at the farm would be healthy. The test mix
also limits how precisely I can measure the rates: with only 70 healthy images,
each wrongly rejected healthy avocado moves the false-alarm rate by 1.4 points.

**The 0.80 mAP will probably not hold on the farm; I treat it as an upper
bound.** The test images look like the training images (same sources, same
studio and supermarket style), not like avocados on my family's sorting table,
with different lighting, dirt, backgrounds and possibly varieties. I also found
at least one test image that is the same photo as a training image, slightly
cropped, which inflates the score. Real miss rates are likely higher, so real
costs are higher too. Before relying on any of these numbers I would run a
small pilot: photograph a few hundred avocados on the actual sorting table,
label them, and measure again.

## Finetuning and baselines

**Model and task.** I finetuned `yolo11n` for **object detection** rather than
image classification, for three reasons:

- A photo can hold several avocados. Most test images show one, but some show
  up to ten. A classifier gives one label per image, so a tray with one diseased
  fruit among healthy ones would get a single label that doesn't say which fruit
  is the problem. A detector boxes and labels each one.
- The box shows the sorter where to look. A fruit sent for review can be
  confirmed or overruled in a few seconds because the reviewer goes straight to
  the boxed spot. My 6-seconds-per-check assumption depends on this.
- Boxes make mistakes visible. If the model flags a fruit because of a shadow
  or a sticker, the box shows it; a bare "diseased" label would hide it.

The trade-off is that detection needs box annotations, which are more work to
produce than image labels. Here the dataset already has them.

**Training.** On my own MacBook (Apple M5 Pro, GPU via MPS), starting from the
pretrained COCO `yolo11n.pt` checkpoint: 40 epochs, image size 640, batch 16,
early-stopping patience 10, seed 0. It took 2.3 hours. The best epoch (by
Ultralytics' validation fitness score, mainly mAP50-95) is saved as `weights/avocado_yolo11n.pt`.

**Results on the same 268-image test split.** For the decision, each image gets
one score: the highest confidence of any disease box. The cost column uses my
Ecuador figures and each model's own best single threshold.

| Model | mAP50 | mAP50-95 | Diseased-vs-healthy AUC | Best cost per 1,000 avocados |
|---|---|---|---|---|
| **Finetuned yolo11n** | **0.800** | **0.526** | **0.987** | **€36.66** |
| COCO yolo11n, zero-shot | 0 | 0 | 0.46 | €288 (best it can do is reject everything) |
| Trivial: predict nothing (export everything) | 0 | 0 | 0.50 | €800 |
| Trivial: majority class of the test split (reject everything) | 0 | 0 | 0.50 | €288 |

The COCO model has no avocado or disease classes, so it cannot name a lesion
however good its general vision is; its mAP on my classes is zero by
definition. For its decision score I used the highest confidence of anything it
detects. That gives an AUC of 0.46, which is essentially chance (0.5 means no
signal), so it is no better than the trivial baselines. It is a floor: what I
would get without finetuning. Finetuning works this well because the network
already knows general visual features (edges, textures, shapes) from
pretraining and only has to learn the avocado-specific classes. The gap between
finetuned and COCO is real, but as said above, the absolute 0.80 is measured on
images like the training data and may not carry over to the farm.

The gap is about what I expected: I thought finetuning on a few thousand
labelled photos would clearly beat a model that has never seen these classes.
What surprised me more was how completely the COCO model fails. I expected a
strong general model to pick up at least some signal, but with an AUC of 0.46
it does no better than chance, which shows that general object recognition
does not transfer to telling a diseased avocado from a healthy one.

## Per-class evaluation and failure modes

**Positive class:** diseased (any of the four diseases). Per class, on the test
split:

| Class | Test boxes | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|
| Cercospora Spot | 114 | 0.84 | 0.82 | **0.86** | 0.68 |
| Healthy | 103 | 0.73 | 0.80 | 0.85 | 0.64 |
| Sunblotch | 26 | 0.82 | 0.89 | 0.83 | 0.41 |
| **Anthracnose** | 133 | **0.67** | **0.61** | **0.66** | **0.37** |
| Stem End Rot | 0 | n/a | n/a | n/a | n/a |
| **All (mAP)** | 376 | | | **0.80** | **0.53** |

Confusion matrix at confidence 0.25 (each row is the true class, each column
what the model predicted; "missed" means no box at all):

| True ↓ / Predicted → | Anthracnose | Cercospora | Healthy | Sunblotch | Missed |
|---|---|---|---|---|---|
| Anthracnose (133) | **84** | 6 | **7** | 1 | **35** |
| Cercospora Spot (114) | 4 | **93** | 0 | 3 | 14 |
| Healthy (103) | 3 | 1 | **84** | 0 | 15 |
| Sunblotch (26) | 0 | 0 | 0 | **23** | 3 |
| No object (false boxes) | 37 | 10 | 28 | 1 | |

**Cercospora Spot carries the headline number**: large, clearly visible spots,
the best AP50 and very few confusions. **Anthracnose drags it down.** It is
the most common disease in the test set, yet the model misses about a quarter
of its lesions entirely (35 of 133) and calls 7 of them Healthy, which is
the dangerous confusion for my decision. It also produces the most false boxes
(37), so the model is both missing real Anthracnose and seeing it where there
is none.

**Why Anthracnose is hardest: the lesions it misses are small.** I checked the
missed cases rather than assume it. The Anthracnose lesions the model found
cover a median of 25% of the image; the ones it missed cover 6.5%. Almost half
of the missed lesions (46%) are smaller than 5% of the image, against 7% of the
found ones. After the image is shrunk to 640 pixels, a lesion that small is only
a handful of pixels, and an early dark spot can look like an ordinary bruise or
blemish. Small boxes are also scored harshly, because a few pixels of offset is
enough to fail the overlap rule.

**At the level of my decision** (one score per image), the misses look
different. At the single threshold of 0.16, 5 of 198 diseased images are
exported. Every one of them involves Anthracnose, and none is a clean
single-fruit photo: one is a research-paper figure, one a collage, two are
stock photos of piles with many avocados, and one is a phone photo of a fruit
held in a hand. So the model fails on small lesions and on images unlike its
typical training photo.

**What this means for the farm.** The model's weakest class is exactly where an
error costs most: a missed diseased avocado costs €20, about 65 times the €0.30
of rejecting a healthy one. Even with the review band, about half of the
remaining cost (roughly €8 of the €16 per 1,000) comes from the 1% of diseased
avocados that still get exported. To reduce it I would photograph closer or at
higher resolution so small spots cover more pixels, collect more labelled
Anthracnose examples, especially early spots that look like bruises, and
consider a lower threshold for Anthracnose boxes specifically, so borderline
cases go to a person, since a check costs under a cent (6 s at €3.50/h). The
domain-shift warning applies most here: dirt, dust and poor light on a real
sorting table look exactly like small dark spots, so the farm's miss rate for
Anthracnose could be worse than on the test set.

## Cost matrix and operating point

All figures are my own assumptions for a small Ecuadorian producer, stored in
`costs.json`. I have no records from my family's farm, so each one is an
estimate with its reasoning.

| Actual ↓ / Decision → | Export | Human review | Reject |
|---|---|---|---|
| **Diseased** | **€20** (FN) | €0.006 check + 3% × €20 = €0.61 | €0 (TP) |
| **Healthy** | €0 (TN) | €0.006 check + 3% × €0.30 = €0.015 | **€0.30** (FP) |

- **FN, €20.** The avocado itself is worth well under a euro, but rot spreads to
  the fruit around it in the box, and the buyer can discount or reject the
  shipment. €20 is my judgement of that risk per missed fruit, not a measured
  number. It is the figure I am least sure of.
- **FP, €0.30.** A rejected healthy avocado is not lost: it sells on the local
  market instead. The cost is the difference between the export price and the
  local price per fruit, estimated from Ecuadorian (lower than European) prices.
- **TP and TN, €0.** These are the correct outcomes and I count them as
  costing nothing relative to each other.
- **Human check, €0.006.** 6 seconds to look at one flagged avocado, at about
  €3.50 per hour, based on Ecuador's minimum wage (about $470 per month in 2025)
  plus employer contributions and the 13th and 14th salaries.
- **Human accuracy, 97%.** A person at the sorting table tires over a shift and
  will not catch every case.
- **Base rate, 4%.** I assume 4% of avocados arriving at the sorting table are
  diseased. Real harvests are mostly clean, unlike the test split (74%
  diseased), so all costs are reweighted to 4%.

**The sweep.** I swept the reject threshold from 0 to 1 in steps of 0.01 and
computed the expected cost per 1,000 avocados at each one:

![Cost against threshold](artifacts/cost_curve.png)

The minimum is at **0.16**, costing **€36.66 per 1,000**. It catches 97.5% of
diseased avocados (5 of 198 exported) and wrongly rejects 5.7% of healthy ones
(4 of 70). The threshold is low for two reasons. A missed diseased avocado
costs about 65 times more than a wrongly rejected healthy one, so it pays to
reject on weak evidence. And the model separates the two classes sharply: only
4 of 70 healthy images score above 0.16, so a low bar costs few false alarms.
Above about 0.6 the cost climbs steeply, because diseased avocados start being
exported.

**The assumption that would most change my answer is the €20 FN cost.** I
re-ran the analysis at other values. The design I chose (review band 0.01 to
0.58, 21% to review) stays the cheapest for any FN cost between about **€8 and
€25**. Below €8, the band narrows and only 12% goes to review. Below about
€2.50, the single threshold jumps from 0.16 to 0.58 and review falls to 6%.
Above about €30, it becomes cheaper to send almost everything (97%) to a
person. That last jump rests on just 2 diseased test images scoring below
0.01, so it is fragile. Because €20 sits inside the €8–25 range, my decision
does not depend on the exact figure, only on a missed fruit costing somewhere in
that range.

## What a person still has to check

**Two thresholds, three outcomes:**

- disease score **≥ 0.58 → reject**
- **0.01 to 0.58 → send to a person**
- **< 0.01 → export**

These are the pair that costs least when I search every combination, with
reviewed fruit costing €0.006 plus the person's 3% error rate. On the test
split, this band rejects 156 diseased and 0 healthy avocados, sends 40 diseased
and 15 healthy to a person, and exports 55 healthy and 2 diseased.

**Load.** Reweighted to a 4% base rate, **21.4%** of avocados go to review. At
5,000 avocados a day that is about **1,070 checks**; at 6 seconds each that is
**1.8 hours a day**, about **a quarter of one person's 7-hour working day**.
That is realistic for my family's farm. A worker who is already at the sorting
table can check the flagged fruit part-time, so no new hire is needed. At peak
harvest, volume may rise, but even doubling it needs only half a person.

**Priced against the alternatives** (€ per 1,000 avocados, at a 4% base rate):

| Policy | € per 1,000 | Comment |
|---|---|---|
| Do nothing (export everything) | €800 | All 40 diseased avocados shipped at €20 each |
| Reject everything | €288 | All 960 healthy avocados sold locally at a €0.30 loss |
| Person checks everything | €38.47 | 8.3 hours a day, about 1.2 people; 3% of disease still missed |
| Model decides alone (threshold 0.16) | €36.66 | No labour, but 2.5% of diseased exported |
| **Model + human review (0.01 / 0.58)** | **€16.03** | Quarter of a person a day |

With Ecuadorian wages, a person checking everything is barely more expensive
than the model alone, so the model's value here is not replacing labour. It is
combining the two: the model handles the clear cases at both ends (78% of
avocados) and a person handles only the uncertain middle. That more than halves
the cost of either on its own. About half of the remaining €16 comes from the
2 diseased avocados in 198 that still score below 0.01 and get exported, all
Anthracnose cases.

**Would I ship it?** Yes, but only after a pilot. The design is clearly
cheaper than the alternatives on these numbers, but the numbers come from photos
unlike my family's sorting table, and Stem End Rot was never tested. Before
relying on it I would photograph and label a few hundred avocados on the actual
sorting table, re-measure the miss and false-alarm rates, and re-run the same
cost sweep with them.

## Limitations & next steps

1. **The photos are not from a sorting table.** The dataset mixes studio shots,
   supermarket photos, stock images and research-paper figures. Every number
   here (0.80 mAP, 97.5% recall, €16 per 1,000) is measured on images like
   that, so it is an upper bound for the farm. *Fix:* photograph and label a
   few hundred avocados on my family's sorting table, under its real lighting,
   and use them as the test set. Then re-run the same cost sweep.
2. **Stem End Rot is untested.** About 7 training photos and none in
   validation or test, so I have no evidence the model catches it. *Fix:*
   collect labelled stem-end-rot photos. Until then, any fruit a sorter suspects
   of it goes to a person regardless of the model.
3. **Anthracnose misses small lesions.** Missed lesions have a median size of
   6.5% of the image, against 25% for found ones, and every diseased avocado
   the model exports involves Anthracnose. *Fix:* photograph closer or at
   higher resolution (or train at a larger image size than 640), and add more
   labelled early-stage Anthracnose, especially spots that look like bruises.
4. **At least one test image duplicates a training image.** I found a test
   photo that is the same as a training photo, only slightly cropped, and two
   automatic checks I tried (image hashing and embeddings) could not reliably
   find others. Test scores may therefore be a little optimistic. *Fix:* split
   the original images myself by source, so near-copies cannot land on both
   sides, and retrain.
5. **The test split is small for the rates that matter.** With 70 healthy
   images, one wrongly rejected avocado moves the false-alarm rate by 1.4
   points, and the jump to reviewing 97% of fruit at high FN costs rests on
   just 2 diseased images. *Fix:* a larger test set, and confidence intervals
   on the miss and false-alarm rates (for example by bootstrapping the test
   images) before choosing thresholds.
6. **The costs are assumptions, not farm data.** I have no records of what a
   missed diseased avocado actually costs my family, or what share of the
   harvest is diseased. *Fix:* ask my family to log, for one season, rejected
   boxes, buyer claims and local-market prices, then replace the €20, €0.30 and
   4% with measured values. The sensitivity check shows the design holds for an
   FN cost between about €8 and €25, so this mainly matters if the real figure
   is outside that range.
7. **The model scores the whole photo, not each avocado.** My decision uses the
   highest disease score in an image, so in a photo of several avocados one
   diseased fruit flags the whole image. On a sorting table each avocado would
   be photographed on its own, so the decision would be per fruit, as intended.
   *Fix:* in the pilot, take one photo per avocado.

## Generative AI use disclosure

I used Claude (Anthropic, through Claude Code) throughout this assignment. It
searched Roboflow Universe and suggested candidate datasets, downloaded the
data and computed the class counts, and wrote the code: the training and
evaluation pipeline (`train.py`), the cost model (`decision.py`) and the Gradio
dashboard (`app.py`). It also ran the analyses I asked for (the missed
Anthracnose cases, the sensitivity of the design to the FN cost) and turned my
notes and answers into the wording of this report, and formatted it.

I also used a separate AI chat to find reference prices and wage figures
(FreshPlaza, FAOSTAT, Eurostat) and to suggest ranges for the costs, and to
draft rough notes on the questions in each section, which I reviewed and
adapted.

The decisions are mine: framing the problem around my family's avocado farm in
Ecuador, choosing the rot and disease dataset, treating any disease as the
positive class, the cost figures and the 4% base rate, which assumption I
consider most uncertain, whether the review load is realistic, and that I
would only ship this after a pilot. I checked the results against the
dashboard and can explain each number in this report.
