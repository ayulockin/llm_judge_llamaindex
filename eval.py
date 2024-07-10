import weave

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator, EvaluationResult
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

import asyncio
from ragas.llms.json_load import json_loader

from rag import SimpleRAGPipeline

WANDB_PROJECT = "llamaindex-weave"
weave.init(f"{WANDB_PROJECT}")

SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a a technical support chatbot.

You are given the following information:
- a user query,
- a reference answer
- a generated answer
- retrieved context for generating the answer

Your job is to judge the relevance and correctness of the generated answer.
Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.

Provide your reasoning for the score you give based on the following guidelines:
- Use the user query
- Use the generated answer
- Use the retrieved context
- Do not use the reference answer to judge the correctness of the generated answer

Output your final verdict by strictly following JSON format:
{{
    "reason": <<Provide a brief explanation for your decision here>>,
    "score": <<Provide a score as per the above guidelines>>,
    "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
}}
"""

### Aligning system prompt using better instructions
# SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a a technical support chatbot.

# You are given the following information:
# - a user query,
# - a reference answer
# - a generated answer
# - retrieved context for generating the answer

# Your job is to judge the relevance and correctness of the generated answer.
# Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.

# Provide your reasoning for the score you give based on the following guidelines:
# - Use the user query
# - Use the generated answer
# - Use the retrieved context
# - Do not use the reference answer to judge the correctness of the generated answer

# Follow these guidelines for scoring:
# - Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.
# - If the generated answer is not correct in comparison to the reference, you should give a score of 1.
# - If the generated answer is correct in comparison to the reference but contains mistakes, you should give a score of 2.
# - If the generated answer is correct in comparision to the reference and completely answer's the user's query, you should give a score of 3.

# Output your final verdict by strictly following JSON format:
# {{
#     "reason": <<Provide a brief explanation for your decision here>>,
#     "score": <<Provide a score as per the above guidelines>>,
#     "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
# }}
# """

### Aligning system prompt using few shot examples
# SYSTEM_TEMPLATE = """You are a Weight & Biases support expert tasked with evaluating the correctness of answers to questions asked by users to a a technical support chatbot.

# You are given the following information:
# - a user query,
# - a reference answer
# - a generated answer
# - retrieved context for generating the answer

# Your job is to judge the relevance and correctness of the generated answer.
# Your score has to be between 1 and 3, where 1 is the worst and 3 is the best.

# Provide your reasoning for the score you give based on the following guidelines:
# - Use the user query
# - Use the generated answer
# - Use the retrieved context
# - Do not use the reference answer to judge the correctness of the generated answer

# Here are a few examples for you as reference:

# 'Question: \nI\'m getting Internal Server Error for url: https://api.wandb.ai/graphql when trying to download an artifact with the following code \n```\nprocessed_data_at = run.use_artifact(f\'params.PROCESSED_DATA_AT:latest\')\nprocessed_dataset_dir = Path(processed_data_at.download())\n```\n Reference Answer: \nAn "Internal Server Error" typically indicates that there\'s a problem on the server side of the application you\'re trying to interact with—in this case, the W&B (Weights & Biases) API. Here are some steps you can take to troubleshoot and potentially resolve the issue:\n\n1. **Check W&B Status**: Before diving into debugging, check if W&B is experiencing any known issues by visiting their status page. If there\'s a known outage or issue, it might be the cause of the error, and you\'ll have to wait until it\'s resolved.\n\n2. **Retry the Request**: Sometimes, intermittent server issues can cause this error. Try running your code again after a short wait to see if the issue resolves itself.\n\n3. **Review Your Code**: Double-check the artifact name and ensure that it\'s correctly formatted and exists in your W&B project. The `params.PROCESSED_DATA_AT` variable should contain the correct artifact name.\n\n4. **API Limits**: If you\'re making a lot of requests in a short period, you might be hitting rate limits. Review the W&B documentation to see if there are any rate limits that you might be exceeding.\n\n5. **Authentication**: Ensure that you\'re properly authenticated with W&B. If your authentication token is missing or incorrect, it could lead to errors. You can re-authenticate using the `wandb login` command.\n\n6. **W&B Version**: Check if you\'re using an outdated version of the W&B client library. If so, update to the latest version using `pip install --upgrade wandb`.\n\n7. **Error Logs**: Look for any additional error messages or logs that might provide more context about the error. Sometimes, the W&B client will output more detailed error information in the console.\n\n8. **Contact Support**: If none of the above steps work, reach out to W&B support with details of the error, including the full traceback and any relevant logs. They might be able to provide more specific guidance or information about the issue.\n\nHere\'s an example of how you might handle exceptions and retry the download in your code:\n\n```python\nfrom pathlib import Path\nimport wandb\nimport time\n\n# Initialize your W&B run\nrun = wandb.init()\n\n# Attempt to download the artifact with retries\nmax_retries = 3\nfor attempt in range(max_retries):\n    try:\n        processed_data_at = run.use_artifact(f\'params.PROCESSED_DATA_AT:latest\')\n        processed_dataset_dir = Path(processed_data_at.download())\n        break  # If the download is successful, break out of the loop\n    except Exception as e:\n        print(f"Attempt attempt+1 failed with error: e")\n        if attempt < max_retries - 1:\n            time.sleep(2**attempt)  # Exponential backoff\n        else:\n            raise  # Re-raise the exception if all retries fail\n\n# Close the W&B run\nrun.finish()\n```\n\nRemember to replace `params.PROCESSED_DATA_AT` with the actual variable or string that contains your artifact name.\n Generated Answer: \nThe "Internal Server Error" you\'re encountering might be due to several reasons. Here are a few things you can check:\n\n1. **Artifact Name**: Ensure that the artifact name you\'re using in `run.use_artifact(f\'params.PROCESSED_DATA_AT:latest\')` is correct and exists in your W&B project.\n\n2. **Bucket Configuration**: If your artifact is stored in an external bucket (like AWS S3, GCP, or Azure), make sure that the CORS configuration of your bucket allows `app.wandb.ai`. Also, if your bucket has object versioning enabled, W&B will retrieve the object version corresponding to the state of the file at the time an artifact was logged.\n\n3. **Credentials**: W&B uses the default mechanism to look for credentials based on the cloud provider you use. Make sure your credentials are correctly set up.\n\n4. **Server Issues**: The error might be due to temporary server issues. You can try again after some time.\n\nIf the issue persists, it would be helpful to provide more details or contact W&B support for further assistance.\n Score: \n1\n Feedback: \nThe generated answer addresses several potential causes for the \'Internal Server Error\' and provides actionable steps to troubleshoot the issue. It mentions checking the artifact name, bucket configuration, credentials, and potential server issues, which are relevant to the user\'s problem. However, it could be more comprehensive by including steps like checking W&B status, retrying the request, reviewing API limits, and ensuring proper authentication, which are also important troubleshooting steps.\nQuestion: \ni am trying to figure out the best way to organize/name/group my wandb runs when doing hyperparam tuning with cross-validation.\n\nideally, i\'d like to be able to see the average score across all the runs that are part of a CV set. and then see the best CV score across the whole hyperparam sweep. (i am using optuna, not wandb, to manage the hyperparam sweep)\n\nis there a recommended approach for this?\n Reference Answer: \nWhen using W&B (Weights & Biases) in conjunction with Optuna for hyperparameter tuning with cross-validation, you can organize your runs effectively by leveraging W&B\'s features such as projects, groups, tags, and notes. Here\'s a recommended approach to achieve your goals:\n\n1. **Projects**: Use a dedicated project for your hyperparameter sweep. This will allow you to compare all runs within the same context.\n   ```python\n   wandb.init(project="your_hyperparam_sweep_project")\n   ```\n\n2. **Groups**: Group your runs by the hyperparameter configuration. Each group can represent one complete set of cross-validation folds for a particular hyperparameter set. This way, you can easily compare different configurations.\n   ```python\n   wandb.init(group=f"config-config_id")\n   ```\n   Here, `config_id` could be a unique identifier for each hyperparameter configuration generated by Optuna.\n\n3. **Tags**: Use tags to mark runs with specific characteristics, such as the fold number or whether it\'s part of the best-performing CV set.\n   ```python\n   wandb.init(tags=[f"fold-fold_number", "best_cv_score"])\n   ```\n\n4. **Notes**: Add notes to each run to describe any particularities or observations that might be relevant when reviewing the results.\n\n5. **Reports**: Create reports to summarize the findings of your hyperparameter sweep, including tables or graphs that show the average score across all CV sets and the best CV score.\n\n6. **Custom Metrics**: Since you want to see the average score across all runs in a CV set and the best CV score across the whole sweep, you can calculate these metrics manually and log them as custom metrics in W&B.\n   ```python\n   wandb.log("average_cv_score": avg_score, "best_cv_score": best_score)\n   ```\n\n7. **Result Aggregation**: To see the average score across all runs that are part of a CV set, you can use W&B\'s grouping feature in the UI to aggregate results. For example, you can group by the hyperparameter configuration and then average the performance metric across the folds.\n\n8. **Analysis**: To find the best CV score across the entire hyperparameter sweep, you can use W&B\'s query and filtering capabilities in the UI to sort the runs by performance and select the top-performing configurations.\n\nBy following these steps, you can effectively organize and analyze your runs in W&B while using Optuna for the hyperparameter sweep. This setup will help you track and compare the performance of different hyperparameter sets and cross-validation folds, leading to more informed decisions about model selection.\n Generated Answer: \nTo organize/name/group your W&B runs when doing hyperparameter tuning with cross-validation, you can use the following approach:\n\n1. **Define the Sweep:** Create a dictionary or a YAML file that specifies the parameters to search through, the search strategy, the optimization metric, etc. This is where you define the hyperparameters to sweep over and the type of sweep to use.\n\n2. **Initialize the Sweep:** Use the `wandb.sweep(sweep_config)` command to initialize the sweep.\n\n3. **Run the Sweep Agent:** Use the `wandb.agent(sweep_id, function=train)` command to run the sweep agent.\n\nFor each run, you can log the average score across all the runs that are part of a CV set using `wandb.log("average_score": average_score)`. \n\nTo visualize the results, W&B provides automatic generation of useful plots like the Parallel Coordinates Plot and the Hyperparameter Importance Plot. The Parallel Coordinates Plot maps hyperparameter values to model metrics, which is useful for honing in on combinations of hyperparameters that led to the best model performance. The Hyperparameter Importance Plot shows which hyperparameter values had the biggest impact on your metrics.\n\nTo see the best CV score across the whole hyperparam sweep, you can use the W&B dashboard to sort your runs by the metric of interest and find the run with the best score. \n\nRemember, both the sweep and the run must be in the same project. Therefore, the name you provide when you initialize W&B must match the name of the project you provide when you initialize a sweep.\n Score: \n3\n Feedback: \nThe generated answer provides a general approach to organizing W&B runs during hyperparameter tuning with cross-validation. It mentions defining the sweep, initializing it, and running the sweep agent, which are relevant steps. However, it lacks specific details on using W&B features like projects, groups, tags, and notes to organize runs effectively. It also does not address how to group runs by hyperparameter configuration or how to manually calculate and log custom metrics for average and best CV scores.\n'

# Output your final verdict by strictly following JSON format:
# {{
#     "reason": <<Provide a brief explanation for your decision here>>,
#     "score": <<Provide a score as per the above guidelines>>,
#     "decision": <<Provide your final decision here, either 'correct', or 'incorrect'>>
# }}
# """

USER_TEMPLATE = """
## User Query
{query}

## Reference Answer
{reference_answer}

## Retrieved Context
{contexts}

## Generated Answer
{generated_answer}
"""

EVALUATOR_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_TEMPLATE),
        ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE),
    ]
)


async def safe_parse_eval_response(eval_response, passing_decision):
    try:
        eval_response_dict = await json_loader.safe_load(eval_response, llm=OpenAI())
        score = eval_response_dict.get("score")
        reasoning = eval_response_dict.get("reason")
        decision = eval_response_dict.get("decision") == passing_decision

    except Exception as e:
        print(e)
        print(eval_response)
        score = 0
        reasoning = "Unable to parse response"
        decision = False
    return decision, reasoning, score


class WandbCorrectnessEvaluator(CorrectnessEvaluator):
    async def aevaluate(
        self,
        query=None,
        response=None,
        contexts=None,
        reference=None,
        sleep_time_in_seconds: int = 0,
        **kwargs,
    ) -> EvaluationResult:
        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None or reference is None:
            print(query, response, reference, flush=True)
            raise ValueError("query, response, and reference must be provided")

        eval_response = await self._llm.apredict(
            prompt=self._eval_template,
            query=query,
            generated_answer=response,
            reference_answer=reference,
            contexts=contexts,
        )

        passing, reasoning, score = await safe_parse_eval_response(
            eval_response, "correct"
        )

        return EvaluationResult(
            query=query,
            response=response,
            passing=passing,
            score=score,
            feedback=reasoning,
        )


@weave.op()
async def get_answer_correctness(
    question: str, answer: str, model_output: dict
) -> dict:
    result = await correctness_evaluator.aevaluate(
        query=question,
        response=model_output["response"],
        reference=answer,
        contexts=model_output["context_str"],
    )
    return {
        "answer_correctness": result.dict()["passing"],
        "feedback": result.dict()["feedback"],
    }


@weave.op()
def compare_length_within_95_percentile(answer: str, model_output: dict) -> dict:
    generated_answer = model_output["response"]

    ground_truth_length = len(answer)
    generated_length = len(generated_answer)

    lower_bound = 0.95 * ground_truth_length
    upper_bound = 1.05 * ground_truth_length

    return {
        "length_within_95_percentile": lower_bound <= generated_length <= upper_bound
    }


@weave.op()
def check_code_block_presence(answer: str, model_output: dict) -> dict:
    generated_answer = model_output["response"]

    has_code_block_gt = "```" in answer
    has_code_block_gen = "```" in generated_answer

    return {
        "code_block_presense": has_code_block_gt == has_code_block_gen,
        "has_code_block_gt": has_code_block_gt,
        "has_code_block_gen": has_code_block_gen,
    }


@weave.op()
def check_bullet_points_presence(answer: str, model_output: dict) -> dict:
    generated_answer = model_output["response"]

    def contains_bullet_points(text):
        bullet_points = ["-", "*", "•"]
        return any(bullet in text for bullet in bullet_points)

    has_bullet_points_gt = contains_bullet_points(answer)
    has_bullet_points_gen = contains_bullet_points(generated_answer)

    return {
        "bullet_points_presense": has_bullet_points_gt == has_bullet_points_gen,
        "has_bullet_points_gt": has_bullet_points_gt,
        "has_bullet_points_gen": has_bullet_points_gen,
    }


judge_llm = OpenAI(model="gpt-4o")

correctness_evaluator = WandbCorrectnessEvaluator(
    llm=judge_llm,
    eval_template=EVALUATOR_TEMPLATE,
)

eval_dataset = weave.ref("eval_data_subset").get()

evaluator = weave.Evaluation(
    dataset=eval_dataset,
    scorers=[
        get_answer_correctness,
        compare_length_within_95_percentile,
        check_code_block_presence,
        check_bullet_points_presence,
    ],
)

if __name__ == "__main__":
    rag_pipeline = SimpleRAGPipeline(similarity_top_k=10)
    rag_pipeline.build_query_engine()
    asyncio.run(evaluator.evaluate(rag_pipeline))
