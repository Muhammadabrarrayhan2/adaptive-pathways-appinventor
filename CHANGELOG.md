# Changelog

## 0.2.0 (2026-10-22) - recommender

Two changes. Both were measured on all 396 analysed projects before the RQ2 evaluation set was frozen; no reviewer
had rated anything yet.

* **AI techniques count as new concepts for AI projects.** Concept flags only record whether a project uses *some* AI.
  For a project that already did, every AI curriculum unit therefore looked like "nothing new". The recommender fell
  back to resources from lower stages: in v0.1, 100% of first suggestions for stage-5 projects went backwards, 16 of
  their 24 suggestions went two or more stages back, and no AI deepening was possible. Since v0.2, an AI technique
  the project has not used yet counts as a new concept. The techniques are image/sound classification, pose and face
  tracking, generative text, generative images, voice assistants, reinforcement learning and augmented reality.
  They are detected from resource titles/URLs and project component types (`AI_TECHNIQUES`).
* **Graded penalty for going backwards.** In v0.1, `stage_fit` scored every backward step 0.3. It is now 0.3 for one
  stage back, 0.15 for two and 0.05 for three or more (`STAGE_FIT`).

Effect across the corpus:

* Stages 1-3: top-3 suggestions unchanged for 0 of 323 projects.
* Stage 4: top-3 changed for 8 of 65 projects (first suggestion for 1).
* Stage 5: top-3 changed for 7 of 8 projects (first suggestion for 4).
* Stage-5 projects with an AI curriculum unit in their top 3: 4/8 → 7/8. Suggestions two or more stages back for
  stage-5 projects: 16 → 5.
* The first suggestion for stage-5 projects is still often a foundational gap (e.g. variables in an AI app that has
  none). This is deliberate: it follows the teaching order that the concept-pathway analysis supports (34/34 links).

Not changed, on purpose: QuizMe remains the first suggestion for 121 of 122 stage-2 projects. Every stage-2 project
lacks Lists by definition, and only five catalog resources lead to stage 3. This is a property of the catalog, not
something to tune away with weights.

## 0.1.0 - first archived release (DOI 10.5281/zenodo.23560460)
