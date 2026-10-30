# InterviewJudge.md — 150 Interview Questions for the LLM-as-a-Judge Alignment Study
**As asked by a Senior AI Engineer & Senior AI Architect**

> Covers: RLAIF/RLHF theory, LLM evaluation design, statistical alignment, rubric engineering, Pydantic schema enforcement, heuristic pre-filtering, post-training pipelines, system design, and deployment.

---

## EASY (20 Questions)
*Fundamentals — You should know these cold.*

**1.** What does RLAIF stand for, and how does it differ from RLHF?

**2.** Why would a financial institution prefer RLAIF over RLHF for post-training alignment?

**3.** What is a "probe output" in the context of this project?

**4.** What are the four evaluation axes used in this project? Why were these four chosen specifically for financial text?

**5.** What is Cohen's Kappa, and what does a value of 0.70 indicate in a human–judge alignment study?

**6.** What is the difference between Fleiss's Kappa and Cohen's Kappa? When do you use each?

**7.** What does "Rubric A" represent in this project, and what is its main weakness compared to Rubric B?

**8.** What is Chain-of-Thought (CoT) prompting, and how does it affect judge score quality?

**9.** What is a Type-Token Ratio (TTR), and why is it useful for evaluating fluency?

**10.** What is Jaccard similarity, and how is it used to proxy factual grounding in this project?

**11.** What is a Likert scale, and what scale was used for human annotations here (0–3 or 0–10)?

**12.** What does `@st.cache_data` do in Streamlit? Why is it important for this app?

**13.** What is Pydantic v2, and why is it used in this pipeline instead of plain Python dicts?

**14.** What does the `field_validator` decorator in Pydantic do? Give one example from this project.

**15.** What is a "false positive" in the context of AI judge evaluation? Give a concrete financial example.

**16.** What was the average local scorer latency achieved in this project, and what does it imply for throughput?

**17.** What files are generated in `data/judge_scores/`? What format are they in?

**18.** What is Label Studio, and why is it referenced in this project's annotation pipeline?

**19.** What is a `ground_truth_quality` field in a probe JSON record? What axes does it contain?

**20.** What command launches the Streamlit dashboard for this project?

---

## MEDIUM (30 Questions)
*Applied — You should be able to explain and implement these.*

**21.** Walk me through the end-to-end pipeline from raw probe generation to final statistical report. What are the 6 stages in `run_pipeline.py`?

**22.** Why did the project use Quadratic Weighted Cohen's Kappa instead of simple unweighted Kappa for ordinal alignment measurement?

**23.** Explain how `discretize_scores()` maps continuous [0.0, 1.0] scores to ordinal Likert bins. What could go wrong if this discretization is poorly calibrated?

**24.** The project achieved Fleiss's κ = 0.829 across 3 human annotators. What does this tell us about the difficulty of the annotation task? Is this ceiling high enough to justify RLAIF?

**25.** Rubric B achieved an overall Cohen's κ = 0.747 compared to Rubric A at 0.594. What specific prompt engineering changes drove this improvement?

**26.** Explain the n-gram repetition penalty calculation in `LightweightStudentEvaluator`. Why are tri-grams (n=3) weighted more heavily than bi-grams (n=2)?

**27.** The local scorer achieved r = 0.820 Pearson correlation on Fluency but only r = 0.500 on No Hallucination. Explain architecturally why this gap exists and what it implies for routing policy.

**28.** How does the retry/error feedback loop in `TeacherJudge.evaluate_single()` work? What does the re-query message contain and why?

**29.** A Groq 70B response returns `{"fluency": 95.0}`. How does the Pydantic `clamp_and_round` validator handle this? What would happen without it?

**30.** What is the `research_disagreement_type` field in probe metadata? How is it used differently by Rubric A and Rubric B simulation modes?

**31.** The project identifies 25% of disagreements as "Rubric Ambiguity." What does this mean for ROI calculation when deciding between rubric engineering vs. scaling model size?

**32.** Explain the fast-pass pre-filter routing architecture. At what threshold is a probe escalated to the 70B teacher judge versus approved locally?

**33.** What is Spearman's ρ and how does it differ from Pearson r? Why might Spearman be more appropriate than Pearson for some axes in this study?

**34.** The `generate_probe_dataset()` function creates 5 categories. What are they and what distinct true_score distributions are assigned to each?

**35.** How does the `FailureModeAnalyzer` classify a disagreement between "Judge False-Positive" and "Rubric Ambiguity"? What thresholds and conditions distinguish them?

**36.** What is `avg_human_std` in the failure breakdown, and what numerical threshold triggers the "Rubric Ambiguity" classification?

**37.** The `export_to_label_studio()` function converts internal probe records to Label Studio format. What fields are included in `task["data"]` and why?

**38.** What is `hapax_ratio` in local scorer metrics? What does a high hapax ratio indicate about text quality?

**39.** Explain the Prefilter Tradeoff Curve visualization. What do the X-axis, Y-axis 1, and Y-axis 2 represent? What does the optimal cutoff at 0.75 mean practically?

**40.** Why does the Dockerfile run `python run_pipeline.py` during image build rather than at container startup?

**41.** What is `use_column_width=True` in `st.image()` and why was it needed for Streamlit 1.65?

**42.** In the agreement stats, why are Pearson r values computed between local scores and Rubric B specifically, rather than Rubric A?

**43.** What is the `RubricConfig` Pydantic model used for? What happens if the `prompt_template` field is missing in a rubric YAML file?

**44.** Explain how the correlation matrix heatmap in `visualize.py` is constructed. Which 6 score dimensions are compared?

**45.** What is `response_format={"type": "json_object"}` in the Groq API call? What problem does it solve?

**46.** The pipeline uses `.venv` isolation. Why is this critical for reproducibility on shared research computing environments vs. using system-level Python?

**47.** How does the `_extract_json()` method handle LLM responses that wrap JSON in markdown code fences? What regex pattern is used?

**48.** What causes `UnicodeEncodeError: 'charmap' codec can't encode character '\u03ba'` on Windows, and how was it fixed?

**49.** The human annotation simulator uses different jitter ranges for `rubric_ambiguity` vs. `aligned` probes. What are these ranges, and why were they deliberately set differently?

**50.** What is the purpose of `use_container_width=True` in `st.dataframe()`? How does it affect the Streamlit probe inspector UI on different screen sizes?

---

## HARD (50 Questions)
*Deep expertise — These separate senior engineers from juniors.*

**51.** The project's Fleiss's Kappa implementation takes an (N × k) ratings count matrix as input. Walk me through the complete mathematical derivation: what are p̄, p_j, and p_e?

**52.** Quadratic Weighted Cohen's Kappa uses a weight matrix w_ij = 1 - (i-j)²/(k-1)². For k=4 categories, compute the full 4×4 weight matrix and explain why off-by-two disagreements are penalized more than off-by-one.

**53.** If two raters both give random uniform ratings across 4 categories, what is the theoretical expected value of Cohen's Kappa? Prove it mathematically.

**54.** Why is κ ≥ 0.70 the cited RLAIF viability threshold? What papers established this, and what assumptions does it make about the relationship between inter-annotator agreement and downstream RLHF reward signal quality?

**55.** The project uses mock mode simulation where Rubric B adds +0.02 bias and Rubric A adds ±0.16–0.24 noise. How does this calibration decision affect the statistical validity of the Cohen's Kappa results? Is the simulation internally consistent?

**56.** The local scorer's Jaccard hallucination proxy achieves r = 0.50 Pearson correlation. Design an improved grounding metric using embeddings that could push this to r ≥ 0.80 without requiring a 70B model call. What trade-offs does your design introduce?

**57.** A senior architect says: "Your Fleiss's Kappa calculation will be biased because your simulated annotators share the same base `true_quality` value from probe metadata. This creates artificial agreement inflation." How do you respond? Is this a valid criticism?

**58.** The `compute_cohens_kappa()` function uses a normalized confusion matrix (divided by n). What numerical instability could arise when n is small (e.g., n=5 probes), and how would you guard against it in production?

**59.** Explain the difference between the macro-level Fleiss's Kappa computed per-axis vs. an item-level agreement analysis. Which is more appropriate for this study and why?

**60.** The pipeline generates probe categories with fixed `true_scores` for `judge_false_positive` (no_hallucination = 0.30) but the teacher judge scores it at 0.85. If you were writing a production-grade RLAIF system, how would you use this 0.55 delta to trigger a human review escalation policy?

**61.** The Prefilter Tradeoff Curve shows Cost Reduction vs. Decision Safety. Derive the mathematical relationship between these two variables as a function of the threshold τ. At what τ does the marginal cost of escalation equal the marginal benefit of safety?

**62.** The project's local scorer doesn't handle semantic similarity at all — only lexical overlap. Describe a scenario in financial QA where a semantically correct output has near-zero Jaccard overlap with the reference context, causing a false-negative hallucination flag. How would you architect around this?

**63.** In the Streamlit dashboard, `@st.cache_data` caches `load_data()` across sessions. What happens if a new evaluation run updates `data/agreement_metrics.json` while the app is running? How would you implement cache invalidation for this?

**64.** Walk through the full Pydantic v2 validator chain when a model returns `{"fluency": "0.85"}` (string, not float). What coercions fire, in what order, and what is the final output?

**65.** The `response_format={"type": "json_object"}` parameter forces JSON-only output from Groq. What failure modes remain that this parameter does NOT protect against? Name at least three.

**66.** You've deployed this on Streamlit Cloud with `run_pipeline(mock_mode=True, force_generate=True)` in the load path. What happens to cold-start latency, and what engineering change would you make for a production deployment with 1,000 concurrent users?

**67.** The project uses `random.Random(abs(hash(prompt + text[:25])) % (10**6))` to seed noise in Rubric A simulation. Why is this deterministic seeding important for research reproducibility? What could go wrong if Python's default `random` was used instead?

**68.** Fleiss's Kappa assumes all subjects are rated by the same number of raters and ratings are independent. Which of these assumptions is violated in this project, and how does it affect the validity of the κ = 0.829 figure?

**69.** The probe dataset uses only 5 financial templates repeated across 200 probes. From a statistical perspective, why does this template reuse inflate inter-rater agreement artificially? How would you redesign the dataset to avoid this confound?

**70.** Design a production A/B testing framework to determine whether deploying Rubric B over Rubric A in a live RLHF reward model pipeline actually improves downstream LLM output quality. What are your control conditions, success metrics, and stopping criteria?

**71.** The project achieves 0.048 ms per probe local evaluation. At what probe volume per day does this become the bottleneck versus network I/O to the 70B API, assuming a 250 ms API round-trip? Derive the crossover point.

**72.** The `compute_correlation_metrics()` function checks for constant arrays (all values equal) before computing Pearson r and returns 0.0, 1.0. Why is this check necessary? What would scipy.stats.pearsonr return for a constant input array in Python 3.11?

**73.** Describe how you would implement a Bayesian version of Cohen's Kappa that produces uncertainty intervals around κ rather than a point estimate. Why might this be more useful in a research context?

**74.** The failure taxonomy shows 29.2% of disagreements are Judge False-Positives. In a live RLHF training loop, if the reward signal systematically over-rewards hallucinated outputs, what training failure mode does this induce? Name the specific phenomenon and its long-term effect on model behavior.

**75.** The project commits `data/` to git. What are the data engineering anti-patterns this creates at scale (e.g., 10M probes), and what infrastructure (DVC, S3, Artifact Registry) would you use instead?

**76.** Walk through the statistical test you would use to determine whether the improvement from Rubric A (κ=0.594) to Rubric B (κ=0.747) is statistically significant. What is the appropriate null hypothesis, test statistic, and required sample size for 80% power?

**77.** The `plot_prefilter_tradeoff_curve()` function defines "agreement preservation" as: if the local scorer approves an output AND the teacher judge also scores it ≥ 0.70, it counts as preserved. What is the logical flaw in this definition, and how would you correct it?

**78.** The hallucination proxy uses `min(0.92, max(0.20, jaccard_overlap * 1.5))`. Why is the ceiling set at 0.92 rather than 1.0? What implicit assumption about local evaluation reliability does this cap encode?

**79.** Explain how entity-level hallucination (wrong numbers in financial statements) would score differently than omission hallucination (missing key figures) under the Jaccard overlap proxy. Which is more dangerous in a financial context, and how would you architect a detector for each?

**80.** The project uses `scipy.stats.pearsonr` for correlation. Under what distributional assumptions is Pearson r the correct statistic here? Are these assumptions met given that scores are bounded in [0, 1]? What transformation would improve validity?

**81.** A model generates: *"Q4 NII was $12.9 billion."* The reference context states: *"Net Interest Income was $12.9 billion."* The Jaccard overlap is very high (0.85), but the model hallucinated Q4 (it was actually Q3). How does your current pipeline handle this, and what would it take to catch it?

**82.** In the context of Constitutional AI and RLAIF at Anthropic, how does rubric-based critique-revision differ from the approach used in this project? What are the trade-offs in annotation quality, cost, and alignment consistency?

**83.** The project's disagreement threshold is |Δ| > 0.30. How did you choose 0.30 vs. 0.25 or 0.35? Walk through the mathematical sensitivity analysis. What happens to the failure taxonomy distribution at each threshold?

**84.** The self-healing `load_data()` in the Streamlit app calls `run_pipeline()` on cache miss. Explain the race condition that could occur with multiple simultaneous users hitting this path, and how you'd fix it using either file locking or a startup lifecycle hook.

**85.** The pipeline writes scores to flat JSON. Design a scalable data model using PostgreSQL or a columnar format like Parquet that could handle 100K probes with efficient filtering by `category`, `domain`, `disagreement_type`, and axis score ranges.

**86.** What is the "alignment tax" in the context of RLHF, and how does this project's methodology help quantify it for financial domain LLMs?

**87.** The project generates 5 figures at 300 DPI. In a CI/CD pipeline, how would you implement automated visual regression testing to detect if a code change silently broke the correlation scatter plot without a human reviewing the PNG?

**88.** The failure taxonomy categorizes Judge False-Negatives as the judge being "overly strict." In production, which failure mode (FP or FN) is more dangerous for a financial institution deploying an AI feedback system? Justify your answer with regulatory risk reasoning.

**89.** How would you extend this pipeline to a multi-judge ensemble (e.g., GPT-4, Claude 3.5, Gemini 1.5 Pro) and compute an aggregate agreement score? What consensus mechanism would you use, and how would you handle systematic per-model biases?

**90.** The Streamlit live playground evaluates custom text. If a user submits 10,000 characters of text, the TTR and n-gram calculations run client-side in Python synchronously. What is the UX impact, and how would you architect an async evaluation backend?

**91.** If you had to reduce the Groq 70B teacher evaluation cost by 90% while maintaining κ ≥ 0.70 alignment, describe a knowledge distillation approach that trains a smaller judge model on Rubric B outputs. What training data, loss function, and evaluation protocol would you use?

**92.** The project uses `seaborn` for the correlation heatmap. In a production BI environment for a bank, why would you replace this with an interactive Plotly or Vega-Altair chart, and what specific financial analyst workflows does interactivity enable?

**93.** Explain transfer learning degradation: if you train a reward model on this project's 200 probe dataset and deploy it on a different financial LLM benchmark (e.g., FinanceBench), what distribution shift failure modes would you anticipate?

**94.** The rubric YAML files define `axes` as a list of dicts. Design a JSON Schema that validates a rubric YAML file and catches: missing required prompt_template, invalid axis names, and score anchor values outside [0, 1].

**95.** The project's mock teacher judge uses deterministic hashing to seed noise: `abs(hash(prompt + text[:25]))`. On Python 3.3+, `hash()` is randomized by default with `PYTHONHASHSEED`. How does this affect reproducibility, and what change would you make?

**96.** The project achieves κ = 0.829 human ceiling and κ = 0.747 teacher alignment. Calculate the ceiling-relative alignment efficiency: what percentage of maximum achievable agreement does the teacher judge reach? How does this compare to the literature benchmarks for LLM-as-a-Judge alignment?

**97.** Walk through the `compute_fleiss_kappa()` mathematical derivation for the case where all 3 annotators always agree. Prove algebraically that κ = 1.0. Then prove it approaches a lower bound when raters assign ratings randomly.

**98.** The project's failure analysis shows 48 total disagreements out of 200 probes (24%). Calculate the 95% confidence interval for this disagreement rate using the Wilson score interval method. Is the sample size sufficient to detect a 5% difference in disagreement rates between rubrics?

**99.** Design an adversarial probe generation strategy that specifically targets the weakest axis in this study (No Hallucination, r = 0.500). What prompt patterns would you use to systematically generate probes that expose local scorer false-negatives?

**100.** The project is structured as a research study. Convert it to a production-grade financial AI quality gate deployed in a CI/CD pipeline that runs every time a new LLM fine-tune is pushed. What SLAs, thresholds, alerting, and rollback mechanisms would you define?

---

## TRICKY KNOWLEDGE-BASED (50 Questions)
*These catch candidates who memorized answers without deep understanding.*

**101.** The project reports "Fleiss's Kappa = 0.829 across 3 human annotators" but the annotators are simulated, not real humans. Does this make the κ value meaningless? When is simulated IAA analysis still scientifically valid?

**102.** Cohen's Kappa corrects for chance agreement. If two raters always agree but always assign the same score to everything (e.g., both rate everything 3/3), what is Cohen's Kappa? Is this a good outcome?

**103.** The rubric uses "No Hallucination" as an axis. Strictly speaking, can any statistical metric prove the *absence* of hallucination, or does it only measure *absence of detected* hallucination? What are the epistemological implications for deploying this as a safety gate?

**104.** A colleague says: "Your Pearson r = 0.820 on fluency proves the local scorer can replace the 70B teacher." What is the critical flaw in this reasoning?

**105.** The project's classification puts "Rubric Ambiguity" at 25.0% of all disagreements. A skeptic argues this number is unfalsifiable since you defined the categories yourself. How do you defend the taxonomy's objectivity?

**106.** The mock judge gives `judge_false_positive` cases a hallucination score of 0.85 while humans rate 0.30. This is baked into the simulation. Does this mean the failure analysis results are circular/predetermined, or is there still scientific value?

**107.** The project claims 65% API cost reduction via fast-pass filtering. But the local scorer runs on EVERY probe, including the 35% that get escalated to 70B anyway. Does this change the cost savings calculation? Compute the correct figure.

**108.** The `LightweightStudentEvaluator` filters common stopwords before computing Jaccard. Name three financial domain terms that are high-frequency across all financial documents and would be incorrectly treated as stopwords by a generic English stopword list.

**109.** The project asserts Rubric B "CoT + Exemplars" is better than Rubric A "Zero-Shot." Is this a fair comparison? What confounds exist (temperature, token length, exemplar quality) that make the A/B rubric comparison not fully controlled?

**110.** The pipeline uses Python's `random.Random(seed)` for deterministic probe generation. Why is this NOT cryptographically secure and why does it matter (or not matter) in this research context?

**111.** A probe's `ground_truth_quality` is set by the data generator, not real humans. If this dataset were submitted to a peer-reviewed NLP conference, what reviewers' objection would you anticipate first?

**112.** The Streamlit app uses `@st.cache_data` on `load_data()`. This means data is cached for the lifetime of the server session. If 5 users open the app simultaneously for the first time, how many times does `run_pipeline()` execute?

**113.** The project achieved "18/18 tests passing." Does 100% test pass rate mean the system is correct? What category of bugs does the test suite NOT catch?

**114.** The project uses `min(0.92, ...)` as a ceiling for the hallucination proxy. This means no probe can ever receive a perfect 1.0 grounding score locally. Is this a bug or a feature? Justify your answer.

**115.** The `compute_cohens_kappa()` function handles an edge case where `1 - p_e ≈ 0` and returns 1.0. Is this mathematically correct? What real-world scenario produces p_e ≈ 1.0 and what does κ = 1.0 mean in that context?

**116.** The project classifies "Judge False-Negative" as: judge scored low, human scored high. But in a domain where human annotators are known to be overgenerous (common in financial compliance), who is actually "correct" — the strict judge or the lenient human?

**117.** The pipeline auto-generates 200 probes deterministically using `seed=42`. A researcher runs the pipeline twice on the same machine and gets identical results. Does this mean the results are reproducible? What happens when run on a different Python version?

**118.** The Dockerfile runs `python run_pipeline.py` during image build. If the pipeline fails during build, the Docker image build fails. Is this good or bad practice for a production deployment, and what alternative would you use?

**119.** A Groq API call fails on attempt 1 and 2 due to rate limiting (HTTP 429), then succeeds on attempt 3. The retry loop feeds validation errors back to the model. Is it appropriate to send validation error messages to the API for rate-limit failures? What should the retry logic differentiate?

**120.** The agreement bar chart shows Rubric B outperforms Rubric A on Fluency and Coherence but underperforms on Instruction Adherence. If you were presenting this to a CTO, would you report "Rubric B is better" or "results are mixed"? What does the overall κ number hide?

**121.** The project uses Streamlit for the dashboard. A hedge fund CTO says: "We can't use Streamlit in production — it's not enterprise-grade." What are the legitimate technical concerns, and what would you migrate to?

**122.** The label `"research_disagreement_type"` is in every probe record. This field drives all the simulation logic. If this field were removed from production records, what would happen to the teacher judge's accuracy claims?

**123.** The project's `requirements.txt` pins `pydantic>=2.0.0` rather than an exact version. In a production ML pipeline, why is this a risk, and what would you do differently?

**124.** The project's correlation analysis uses all 200 probes including the simulated disagreement cases. Would removing the 48 disagreement probes increase or decrease the Pearson r between local and teacher scores? Why?

**125.** The `figures/` directory is committed to git. GitHub has a 100MB file size limit and LFS recommendations for binary files. What CI/CD issue would appear at scale if you generate 1,000 figures per experiment?

**126.** The hallucination axis has the highest Fleiss's κ = 0.951 among human annotators but only r = 0.500 correlation with the local scorer. Explain this apparent contradiction — high human agreement + low local correlation.

**127.** The Streamlit live playground evaluates text using `mock_mode=True`. A user pastes a real 10-K excerpt and asks "Is this hallucinated?" What is the correct answer the system gives, and is it accurate?

**128.** The project stores probe IDs as `probe_001`, `probe_002`, etc. At 1 million probes, what indexing problem does this string format create in a database, and what ID scheme would you use instead?

**129.** The `run_pipeline.py` has a `force_generate=False` default. When the Streamlit app calls `run_pipeline(force_generate=True)`, it regenerates all 200 probes from scratch. Why is this problematic for a deployed app where users may be mid-session?

**130.** The failure taxonomy uses `avg_human_std > 0.65` as the trigger for "Rubric Ambiguity." This threshold was chosen manually. What data-driven method would you use to learn the optimal classification boundary?

**131.** The project structure has `annotation/`, `analysis/`, `evaluation/`, and `scripts/` as separate directories. A new engineer asks: "Why not put everything in `src/`?" What is your architectural justification for the current layout?

**132.** The project uses `rng.uniform(-0.10, 0.10)` jitter for aligned probes. This means aligned probes can still occasionally show a delta > 0.30 if scores are near the boundary (e.g., human = 0.85, teacher = 0.75, then with jitter human could be 0.79 and teacher 0.87). Does this create false disagreements? How frequent would they be?

**133.** A senior researcher points out: "Your Cohen's Kappa computation discretizes continuous scores to 4 bins, losing information. You should use a continuous agreement metric like Lin's Concordance Correlation Coefficient (CCC) instead." How do you respond?

**134.** The project deployed on Streamlit Cloud with the self-healing pipeline. On Streamlit Community Cloud, apps are put to "sleep" after inactivity. When a user wakes the app, `load_data()` cache is cleared. What happens on first load, and how long does it take?

**135.** The prompt template in rubric_b.yaml uses `{model_output}`, `{prompt}`, and `{context}` placeholders. If a financial model output contains the literal string `{context}`, what bug appears, and how do you fix it safely?

**136.** The project's `compute_fleiss_kappa()` processes an (N × k) matrix where k is fixed at 4 categories (0–3 Likert). What happens mathematically if one category (e.g., category 0) is never assigned by any annotator? Does Fleiss's Kappa still return a valid result?

**137.** The project uses `seaborn-v0_8-whitegrid` style with a version-safe fallback to `"default"`. In a containerized environment using `matplotlib==3.11.2`, which style name would be active? How would you verify this without running the code?

**138.** The `FailureModeAnalyzer` finds the "primary axis" of disagreement by taking `max(disagreed_axes, key=lambda a: disagreed_axes[a]["delta"])`. What happens if two axes have identical deltas? Is the resulting classification stable or arbitrary?

**139.** The `_mock_evaluate()` method reads `metadata.get("research_disagreement_type", "aligned")`. In production with real Groq API calls, this metadata is never used. Explain precisely why the mock and live code paths are architecturally separate, and what risk this separation introduces.

**140.** The agreement stats compute "overall" as the arithmetic mean of per-axis κ values. Is an arithmetic mean the correct aggregation for Kappa statistics? What would a weighted mean (weighted by axis prevalence) look like, and when would it produce a different result?

**141.** The project reports the self-healing pipeline runs in ~3 seconds on a fresh deployment. On Streamlit Community Cloud, CPU is throttled. What realistic cold-start time estimate would you provide to a user, and what UX affordance would you add?

**142.** The `OptimizationFix.md` claims "65% API cost reduction." This assumes probes that pass the local fast-pass filter don't need teacher evaluation. In a safety-critical financial compliance context, is it acceptable to skip teacher evaluation for "approved" outputs? What regulatory frameworks would prohibit this?

**143.** A probe with `category = "hallucination"` generates high local fluency (r=0.85) because hallucinated text is often grammatically fluent. The fast-pass filter approves it. Explain why high fluency correlation on the local scorer is specifically dangerous for the hallucination detection use case.

**144.** The README badges show `Tests: 18/18 Passed` hardcoded as a static badge. In a real CI/CD pipeline, how would you make this badge dynamic and automatically updated on every push?

**145.** The project's Streamlit tab for "Statistical Figures" just shows PNGs with `st.image()`. A quantitative analyst says "I can't drill down into the data from an image." What Plotly/Altair-based implementation would you build instead?

**146.** The `data/human_annotations.json` has a `target_disagreement_axis` field. Why is this field inappropriate to include in a real annotation dataset sent to human raters? What contamination does it cause?

**147.** The project stores all probe outputs as a JSON list rather than JSONL (newline-delimited JSON). At what file size does this become a memory problem when loading into Python? What is the practical threshold for switching to JSONL or Parquet?

**148.** The `run_pipeline.py` orchestrator does not implement parallelism — all 6 steps run sequentially. Identify the two steps that could be parallelized safely, and estimate the speedup achievable on a 4-core machine.

**149.** The project uses `matplotlib` and `seaborn` for all figures. Both libraries generate static rasterized images. For a financial institution's internal BI dashboard, what argument would you make for SVG vector output rather than PNG? What additional code change is needed?

**150.** The project demonstrates that rubric engineering (prompt + exemplar design) improved κ by +0.153 without changing the underlying 70B model. A product manager says: "This means we should always optimize prompts first before buying better models." What nuanced, architecture-aware response would you give to prevent overgeneralizing this finding?

---

## Answer Tips for Interviews

- For **Easy** questions: answer directly and confidently in 30–60 seconds.
- For **Medium** questions: structure your answer with context → implementation → result. Target 2–3 minutes.
- For **Hard** questions: use whiteboard / pseudocode if available. Show your reasoning process, not just conclusions.
- For **Tricky** questions: the correct answer is often "it depends" or "there's a flaw in the premise" — examiners reward honest intellectual precision over defensive justification.

---

*File generated for portfolio project: LLM-as-a-Judge Alignment Study (2026)*  
*github.com/ankitghoshthecreator/JudgeLLM*
