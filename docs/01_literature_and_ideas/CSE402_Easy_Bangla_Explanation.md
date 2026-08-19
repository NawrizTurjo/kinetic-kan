# CSE-402 প্রজেক্ট: সহজ বাংলায় ৭টি ট্র্যাকের বিস্তারিত মূল কথা
> একদম জটিলতাহীন, সহজ ভাষায় ৭টি প্রজেক্টের পেছনের আইডিয়া, পেপারে কী করা ছিল এবং আমাদের অতিরিক্ত কী কোড করতে হবে।

---

## 🥇 ১. Track B: Bayesian Epidemic Modeling (মহামারীর গতিবিধি ও অজানা মান নির্ণয়)

* **সহজ উদাহরণ:** একটা স্কুলে ফ্লু ছড়িয়ে পড়েছে। প্রতিদিন কতজন অসুস্থ হচ্ছে তার ১৪ দিনের ডেটা আছে। আমাদের কাজ হলো—একজন রোগী থেকে গড়ে কতজন আক্রান্ত হচ্ছে ($R_0$) তা নিখুঁতভাবে বের করা।
* **কতদিন আগের পেপার?** মূল পেপার ২০২১ সালের (*Statistics in Medicine*) এবং এর সাথে ২০২৪ সালের একটি রয়্যাল সোসাইটির পেপার।
* **মূল পেপারে কী করা ছিল?** ১৯৭৮ সালের ইংল্যান্ডের একটি স্কুলের ১৪ দিনের ফ্লু ডেটায় SIR ডিফারেনশিয়াল ইকুয়েশন সলভ করে Stan লাইব্রেরির মাধ্যমে Bayesian MCMC চালিয়ে রোগের প্যারামিটার ($R_0 \approx 3.23$) বের করেছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. আমরা কোনো রেডিমেড লাইব্রেরি ব্যবহার না করে পাইথনে **হাত দিয়ে Metropolis-Hastings MCMC কোড করব**।
  2. প্রমাণ করে দেখাব যে **খারাপ ODE সলভার (Euler) ব্যবহার করলে ভুলের কারণে MCMC আটকে যায়, কিন্তু RK4 সলভার দিলে সুন্দর স্মুথ রেজাল্ট আসে** (এটা স্যারের জন্য সেরা প্রুফ)।
  3. আমাদের মডেলটি **বাংলাদেশের ২০২৩ সালের ডেঙ্গুর আসল ডেটাসেটে** চালিয়ে মহামারী প্রেডিক্ট করব।
* **Numerical Math:** Runge-Kutta (RK4), Metropolis-Hastings MCMC, Least Squares, Random Number Generation।
* **কেন সেরা?** সিলেবাসের সবচেয়ে বেশি টপিক (~৯৫%) কাভার করে এবং সাধারণ ল্যাপটপে ২ সেকেন্ডে রান করে।

---

## 🥈 ২. Track C: LLM Speculative Decoding (চ্যাটজিপিটি বা এলএলএম দ্রুত করা)

* **সহজ উদাহরণ:** বড় LLM দিয়ে প্রতিটা শব্দ জেনারেট করতে দেরি হয়। তাই একটা ছোট ফাস্ট মডেল আগে খসড়া ৪-৫টা শব্দ লিখে ফেলে, আর বড় মডেলটি একবারে দেখে ঠিক থাকলে 'Accept' করে, ভুল হলে 'Reject' করে।
* **কতদিন আগের পেপার?** ২০২৩ সালের পেপার (*Google Research & DeepMind, ICML 2023*)।
* **মূল পেপারে কী করা ছিল?** তারা দেখিয়েছে যে মেট্রোপলিস নিয়মে ($\min(1, P/Q)$) ড্রাফট টোকেন ভেরিফাই করলে মডেলের কোয়ালিটি একটুও নষ্ট না হয়ে ২ থেকে ৩ গুণ দ্রুত টেক্সট জেনারেট করা যায়।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. Kaggle-এ ছোট মডেল (Qwen-0.5B) এবং বড় মডেল (Qwen-3B) লোড করে হাত দিয়ে **Metropolis-Hastings Rejection Sampling** কোড করব।
  2. খসড়া কয়টি শব্দ ড্রাফট করলে সবচেয়ে কম সময় লাগবে—তার ওপর **1D Optimization** করব।
  3. শব্দের অনিশ্চয়তা (Entropy) বুঝে স্বয়ংক্রিয়ভাবে ড্রাফট সাইজ কম-বেশি করার একটি অ্যাডাপটিভ নিয়ম বানাব।
* **Numerical Math:** Monte Carlo Rejection Sampling, Metropolis-Hastings Rule, Inverse-CDF Categorical RNG, Optimization।
* **কেন সেরা?** ২০২৪-২৫ সালের সবচেয়ে হট এআই টপিক (LLM Acceleration)।

---

## 🥉 ৩. Track A: KAN-ODEs (এআই দিয়ে ফিজিক্সের সমীকরণ আবিষ্কার)

* **সহজ উদাহরণ:** বনে বাঘ আর হরিণ আছে। বাঘ বাড়লে হরিণ কমে, হরিণ কমলে বাঘ না খেয়ে মরে। এই সিস্টেমের ভেতরের অজানা সমীকরণটি এআই নিজে নিজে শিখে নেয়।
* **কতদিন আগের পেপার?** ২০২৪ সালের পেপার (*MIT Research, CMAME Journal*)।
* **মূল পেপারে কী করা ছিল?** সাধারণ নিউরাল নেটওয়ার্কের তারে B-Spline কার্ভ বসিয়ে Lotka-Volterra শিকার-শিকারি সিস্টেমের সমীকরণ খুব কম প্যারামিটারে (মাত্র ২৪০টি) শিখেছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. পেপারের ডিফল্ট সলভার সরিয়ে আমাদের সিলেবাসের **Forward Euler বনাম Heun (RK2) বনাম RK4** বসিয়ে দেখব সলভারের ভুলেই এআই ট্রেনিং নষ্ট হয় কি না।
  2. B-Spline-এর জায়গায় **Lagrange ও Newton Polynomials** বসিয়ে তাদের স্টেবিলিটি তুলনা করব।
* **Numerical Math:** Spline Interpolation, Lagrange/Newton Interpolation, Runge-Kutta ODEs, Gradient Optimization।
* **কেন সেরা?** MIT-র লেটেস্ট সায়েন্টিফিক মেশিন লার্নিং (SciML) ফ্রেমওয়ার্ক।

---

## ৪. Track D: Continuous Graph Neural Diffusion - GRAND (গ্রাফে তথ্য ছড়ানো)

* **সহজ উদাহরণ:** ফেসবুকের ফ্রেন্ড নেটওয়ার্ক বা সাইটেশন গ্রাফে কোনো একটা নোডের ইনফরমেশন পুরো নেটওয়ার্কে কীভাবে কন্টিনিউয়াসলি ছড়িয়ে পড়ে (Heat Diffusion)।
* **কতদিন আগের পেপার?** ২০২১ সালের ল্যান্ডমার্ক পেপার (*ICML 2021*)।
* **মূল পেপারে কী করা ছিল?** গ্রাফ নিউরাল নেটওয়ার্কে বেশি লেয়ার দিলে সব নোড একই রকম হয়ে যায় (Over-smoothing)। এটা ঠেকাতে তারা গ্রাফ প্রপাগেশনকে ডিফারেনশিয়াল ইকুয়েশন হিসেবে মডেল করেছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. হাত দিয়ে **QR Method বা Power Method কোড করে গ্রাফের আইগেনভ্যালু (Eigenvalues)** বের করব এবং দেখাব আইগেনভ্যালুর ক্ষয়ের সাথে ওভার-স্মুথিংয়ের কী সম্পর্ক।
  2. গ্রাফের নোড আপডেট করতে **Explicit RK4 সলভার বনাম Implicit LU Decomposition** সলভারের স্পিড ও নির্ভুলতা তুলনা করব।
* **Numerical Math:** Graph Laplacian Eigenvalues (QR/Power Method), Runge-Kutta ODEs, LU Decomposition।
* **কেন সেরা?** গ্রাফ ডিপ লার্নিং ও লিনিয়ার অ্যালজেব্রার দারুণ কম্বিনেশন।

---

## ৫. Track E: 3D Gaussian Splatting with Levenberg-Marquardt (ছবি থেকে নিখুঁত ৩ডি সিন)

* **সহজ উদাহরণ:** একটা বস্তুর কয়েকটা ২ডি ছবি থেকে একদম রিয়েলিস্টিক ৩ডি মডেল বানানো।
* **কতদিন আগের পেপার?** ২০২৪ ও ২০২৬ সালের পেপার (*Technical University of Munich & 3DV 2026*)।
* **মূল পেপারে কী করা ছিল?** ৩ডি সিনে লাখ লাখ ছোট গোলক অপ্টিমাইজ করতে সাধারণ ফার্স্ট-অর্ডার Adam-এর বদলে সেকেন্ড-অর্ডার **Levenberg-Marquardt (LM)** বসিয়ে ২০-৩০% দ্রুত কনভার্জেন্স এনেছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. ছোট একটি সিন্থেটিক ৩ডি সিনে **Adam বনাম Gauss-Newton বনাম Levenberg-Marquardt** তুলনা করে দেখাব কখন গাউস-নিউটন ক্র্যাশ করে আর কীভাবে LM তাকে বাঁচায়।
  2. নরমাল সমীকরণ সমাধান করতে হাত দিয়ে **LU Decomposition** সলভার বসাব।
* **Numerical Math:** Non-linear Least Squares, Levenberg-Marquardt, Gauss-Newton, LU Decomposition।
* **কেন সেরা?** ৩ডি কম্পিউটার ভিশন ও গেমিংয়ের সবচেয়ে ট্রেন্ডি বিষয়।

---

## ৬. Track F: Differentiable Bundle Adjustment (রোবট ও সেলফ ড্রাইভিং কারের দৃষ্টি)

* **সহজ উদাহরণ:** সেলফ ড্রাইভিং গাড়ি চলার সময় তার ক্যামেরা দিয়ে রাস্তার ৩ডি ম্যাপ আর গাড়ির নিজের অবস্থান একসাথে ঠিক করে নেওয়া (SLAM)।
* **কতদিন আগের পেপার?** ২০২৬ সালের পেপার (*IEEE Transactions on Robotics*)।
* **মূল পেপারে কী করা ছিল?** PyTorch-এ রোবট ও ড্রোন ক্যামেরার পজিশনের ভুল ঠিক করার জন্য সেকেন্ড-অর্ডার লিনিয়ার সলভার দিয়ে একটি ডিফারেনশিয়েবল ফ্রেমওয়ার্ক বানিয়েছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. ক্যামেরা খুব কাছাকাছি থাকলে বা পয়েন্ট কম থাকলে ম্যাট্রিক্সের যে রোগ হয় (**Ill-conditioning $\kappa > 10^6$**), তা মেপে দেখাব কখন অ্যালগরিদম ফেইল করে।
  2. সিস্টেম অব লিনিয়ার ইকুয়েশন সলভ করতে **Gauss Elimination বনাম LU Decomposition with Partial Pivoting** তুলনা করব।
* **Numerical Math:** Gauss-Newton, Levenberg-Marquardt, Matrix Conditioning, LU Decomposition, Gauss Elimination।
* **কেন সেরা?** পিওর রোবোটিক্স ও অপ্টিমাইজেশনের ক্লাসিক ও প্রেস্টিজিয়াস কাজ।

---

## ৭. Track G: Diffusion ODE Sampling - STORK (জেনারেটিভ এআই ছবি তৈরি)

* **সহজ উদাহরণ:** Midjourney বা Stable Diffusion যেভাবে হিজিবিজি নয়েজ থেকে স্টেপ বাই স্টেপ সুন্দর ছবি বানিয়ে ফেলে।
* **কতদিন আগের পেপার?** ২০২৫ সালের পেপার (*UCLA Research*)।
* **মূল পেপারে কী করা ছিল?** ডিফিউশন মডেল দিয়ে ছবি বানানোর সময় রিভার্স ODE সলভ করতে একটি বিশেষ 'Stabilized Runge-Kutta' সলভার দিয়ে কম স্টেপে হাই-কোয়ালিটি ছবি জেনারেট করেছে।
* **আমাদের টিমের অতিরিক্ত কাজের স্কোপ কী (Extension Scope)?**
  1. একটি প্রি-ট্রেইনড CIFAR-10 মডেলে কোনো নতুন ট্রেইনিং ছাড়া শুধু স্যাম্পলিং লুপে **Euler বনাম Heun (RK2) বনাম RK4 বনাম Stabilized RK** বসিয়ে ছবি বানাব।
  2. কত কম স্টেপে (NFE) সবচেয়ে ভালো ছবি (FID Score) পাওয়া যায় তা মেপে গ্রাফ প্লট করব।
* **Numerical Math:** Runge-Kutta ODE Solvers, Stiff Differential Equations, Step-Size Error Analysis।
* **কেন সেরা?** ইমেজ জেনারেশনের পেছনের ম্যাথমেটিকাল ইঞ্জিন নিজে কন্ট্রোল করা।

---

## 🧭 স্যারের সাজেস্ট করা Directions-এর সাথে ৭টি ট্র্যাকের ম্যাপিং

অফিশিয়াল নোটিশে স্যার ৩টি সম্ভাব্য ডিরেকশনের কথা বলেছেন। আমাদের ৭টি ট্র্যাকের কোনটি কোন ডিরেকশন কাভার করে তা নিচে দেওয়া হলো:

| ট্র্যাক নম্বর ও নাম | স্যারের কোন Direction কাভার করে? | সহজ ভাষায় কেন এই Direction? |
| :---: | :--- | :--- |
| **🥇 Track B**<br>*(Bayesian Epidemic)* | **Triple Hybrid (১ + ২ + ৩)**<br>*(সবগুলো Direction একসাথে!)* | • **Methodological Extension:** Stan লাইব্রেরির বদলে হাত দিয়ে Metropolis-Hastings কোড করা এবং Euler বনাম RK4-এর প্রভাব বের করা।<br>• **Cross-Domain:** ১৯৭৮ সালের ব্রিটিশ ফ্লু মডেলকে **২০২৩ সালের বাংলাদেশ ডেঙ্গু ডেটাতে** প্রয়োগ করা।<br>• **Comparative Study:** SIR বনাম SEIR মডেলের কার্যকারিতা তুলনা। |
| **🥈 Track C**<br>*(LLM Speculative Decoding)* | **Cross-Domain + Methodological**<br>*(Direction ১ + ২)* | • **Cross-Domain:** স্ট্যাটিস্টিক্যাল ফিজিক্সের **Metropolis-Hastings মেথডকে LLM (Language Model) স্পিডআপে** প্রয়োগ করা।<br>• **Methodological Extension:** টোকেন এন্ট্রপি অনুযায়ী অ্যাডাপটিভ ড্রাফট লেন্থ ($K^*$) অপ্টিমাইজ করা। |
| **🥉 Track A**<br>*(KAN-ODEs)* | **Methodological + Cross-Domain**<br>*(Direction ২ + ১)* | • **Methodological Extension:** পেপারের ডিফল্ট সলভার বদলে **Euler, RK2, RK4** বসানো এবং B-Spline-এর বদলে **Lagrange/Newton পলিনোমিয়াল** বসিয়ে স্টেবিলিটি পরীক্ষা করা।<br>• **Cross-Domain:** শিকার-শিকারি ছাড়াও পেন্ডুলাম বা এপিডেমিক সিস্টেমে প্রয়োগ করা। |
| **Track D**<br>*(Graph Neural Diffusion)* | **Cross-Domain + Comparative**<br>*(Direction ১ + ৩)* | • **Cross-Domain:** ফিজিক্সের হিট ডিফিউশন PDE এবং **Eigenvalue Decomposition (QR)**-কে গ্রাফ নিউরাল নেটওয়ার্কে প্রয়োগ করা।<br>• **Comparative Study:** Explicit RK4 বনাম Implicit LU সলভারের একুরেসি ও ওভার-স্মুথিং তুলনা। |
| **Track E**<br>*(3D Gaussian Splatting LM)* | **Methodological + Comparative**<br>*(Direction ২ + ৩)* | • **Methodological Extension:** 3DGS-এর ডিফল্ট Adam অপ্টিমাইজারের জায়গায় সেকেন্ড-অর্ডার **Levenberg-Marquardt ও LU Solvers** বসিয়ে কনভার্জেন্স দ্রুত করা।<br>• **Comparative Study:** বিভিন্ন নয়েজ লেভেলে Adam বনাম Gauss-Newton বনাম LM তুলনা। |
| **Track F**<br>*(Bundle Adjustment)* | **Comparative + Methodological**<br>*(Direction ৩ + ২)* | • **Comparative Study:** ক্যামেরা বেসলাইন পরিবর্তনের সাথে সাথে ম্যাট্রিক্স কন্ডিশনিং ($\kappa$) এবং Gauss-Newton বনাম LM-এর ব্রেকডাউন পয়েন্ট বের করা।<br>• **Methodological Extension:** লিনিয়ার সলভারে Gauss Elimination বনাম LU with Partial Pivoting তুলনা। |
| **Track G**<br>*(Diffusion ODEs STORK)* | **Comparative + Methodological**<br>*(Direction ৩ + ২)* | • **Comparative Study:** ইমেজ কোয়ালিটি (FID) বনাম স্টেপ সংখ্যায় (NFE) **Euler vs Heun vs RK4 vs Stabilized RK** বেঞ্চমার্ক করা।<br>• **Methodological Extension:** স্টিফ ডিফারেনশিয়াল ইকুয়েশন সলভার দিয়ে স্যাম্পলিং লুপ উন্নত করা। |

> **💡 নিউ ডিরেকশন / হাইব্রিড অ্যাপ্রোচ কী?**  
> স্যার নোটিশে বলেছেন এগুলো রিজিড কোনো নিয়ম নয়। সবচেয়ে বেশি মার্কস পাওয়া যায় যখন প্রজেক্টটি **"Hybrid Direction"** ফলো করে—অর্থাৎ শুধু একটা পেপারের কোড অন্য ডেটায় চালানো নয় (Pure Cross-Domain), বরং পেপারের ভেতরের **Numerical Algorithm নিজে বদলে নতুন কিছু প্রমাণ করা (Methodological Extension) + সেটি একটি বাস্তব বা আধুনিক সিস্টেমে টেস্ট করা (Cross-Domain)**।

---

## 🎯 আপনার জন্য এক লাইনের সিদ্ধান্ত গাইড:
1. **সবচেয়ে নিরাপদ, সহজ কোডিং ও সর্বোচ্চ সিলেবাস কাভারেজ চাইলে:** 👉 **Track B (Epidemic)**
2. **সবচেয়ে আধুনিক ও হাইপড LLM প্রজেক্ট চাইলে:** 👉 **Track C (Speculative Decoding)**
3. **আধুনিক সায়েন্টিফিক এআই (SciML) চাইলে:** 👉 **Track A (KAN-ODEs)**
4. **গ্রাফ ও নেটওয়ার্ক থিওরি পছন্দ হলে:** 👉 **Track D (Graph GRAND)**
5. **৩ডি কম্পিউটার ভিশন বা রোবোটিক্স চাইলে:** 👉 **Track E বা F (3DGS / Bundle Adjustment)**
