[Task]
Your task is to judge a candidate explanation of an xkcd comic in relation to the ground truth explanation given by explainxkcd.com. You should estimate the percentage of the ground truth explanation that is accurately covered by the candidate explanation. No percentage points should be deducted for missing character names, e.g. "Cueball".

[Additional info provided]
This additional info was provided at the time of candidate explanation creation. Just reporoducing this information in the candidate explanation should not count favorably toward that explanation, but accurately ellucidating the meaning of this additional info in relation to the ground truth explanation should count toward a better score.

Title: {{TITLE}}
Mouseover/tooltip text for image: {{MOUSEOVER_TEXT}}

Note that the mouseover/tooltip text is often referred to as "title text" in the ground truth explanations, which may be a little confusing.

[Ground truth]
{{EXPLAIN_XKCD_EXPLANATION}}

[Candidate explanation]
{{CANDIDATE_EXPLANATION}}

[Response Format]
 I am providing a template for how to format your response.
 Failure to respond with the exact response template will result in a parsing error and immediate disqualification.

[Response Template]
<discussion>
  Discuss what the candidate explanation gets right and wrong in relation to the ground truth explanation.
</discussion>
<percentage_covered>
  Give your best estimate for what percentage of the true explanation is correctly covered by the candidate explanation. This should be a whole number, e.g., 0, 10, 45, 80. This section should contain no other information besides that number.
</percentage_covered>