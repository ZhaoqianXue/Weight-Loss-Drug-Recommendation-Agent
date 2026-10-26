# Evaluation questions

Version review-v1 contains the complete 120-question set used to evaluate AskGLP1.

## Files

- [Complete question list](questions.md) is the readable version.
- [questions.tsv](questions.tsv) contains the same 120 records in UTF-8, with a header row and tab-separated fields.

Both files preserve the original IDs, question wording, group assignments, and intended handling. Informal spelling and false premises are retained because they are part of the evaluated inputs.

## Evaluation groups

The fixed groups cover the tasks examined in RQ3. A question can involve several content dimensions while retaining its original group. The files use the readable group names reported in the annotation appendix of the paper.

| Evaluation group | Questions |
|---|---:|
| Counts by drug or brand | 26 |
| Counts by population group | 27 |
| Combinations of details | 14 |
| Supporting quotes | 9 |
| Narratives beside FAERS | 24 |
| Boundary | 20 |

The first five groups contain 100 answerable questions. The boundary group contains 10 questions intended for decline and 10 for a limited answer.

## Fields

| Field | Meaning |
|---|---|
| id | The original question ID, H001 through H120. |
| question | The question exactly as presented for evaluation. |
| evaluation_group | The question's existing evaluation group, shown with its readable label. |
| intended_response | The intended handling, recorded as answer, decline, or limited. |

The intended response is a design label. It is not the only acceptable system response status, the system's observed response, or a human error rating. A response is not judged incorrect solely because its status differs from this label. Human ratings assess the answer against the reference answer using the criteria in the annotation appendix of the paper.

This release provides the evaluation inputs. It contains no user posts or reviews, system answers, or automatic scoring configuration.
