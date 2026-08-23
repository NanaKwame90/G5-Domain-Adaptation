<h1>Language Models Playschool Challenge Leaderboard</h1>
<h3>Welcome to the official leaderboard of the 2026 LM Playschool Challenge!</h3>

In the table below, you will find results - both clemscores and statscores - for a variety of baseline models ran on playpen's eval set (`playpen eval <model_name> --suite all`), to which you can compare the results obtained by your own implementations.

Additionally, the table will be updated with intermediate results submitted by participants in the challenge, so that everyone can keep track of how their competitors are doing.


Team Name | Version ID | Base Model Name | Recipe Description | HF Link | Clemscore | Statscore | Delta Clemscore w.r.t. Base Model | Delta Statscore w.r.t. Base Model |
|---------|------------|-----------------|--------------------|---------|-----------|-----------|-----------------------------------|-----------------------------------|
| LLP: Large Language Problems | 2b-v1 | Qwen/Qwen3.5-2B | LoRA SFT on successful playpen dataset rounds, corrections-based on-policy DPO | No yet available | 54.28 | 42.11 | + 41.23 | - 1.91 |
| bosphorous | iter2 | Qwen/Qwen3.5-2B | LoRA SFT, success-filtered playpen-data |  https://huggingface.co/Shamima/lm-playschool-qwen3.5-2b-sft-iter2 | 46.66 | 40.39 | +33.61 | -3.63 |
|Baseline| /  |Qwen/Qwen3.5-27B |  thinking mode deactivated | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | 65.72 | 65.99 | 0 | 0 |
|Baseline| /  |Qwen/Qwen3.5-9B |  thinking mode deactivated  | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 41.92 | 54.16 | 0 | 0 |
|Baseline| / |Qwen/Qwen3.5-4B | thinking mode deactivated  | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 39.7 | 50.63 | 0 | 0 |
|Baseline| / |Qwen/Qwen3.5-2B |  thinking mode deactivated  | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 13.05 | 44.02 | 0 | 0 |
|Baseline| / |meta-llama/Llama-3.1-8B | thinking mode deactivated  | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 25.67 | 45.81 | 0 | 0 |

<h3>List of teams participating in the challenge</h3>

Here you can find a list of all the teams currently enrolled in the challenge, along with their respective affiliation.


| Name of team<br><br>         | Affiliation of team<br><br>                                     |
| ---------------------------- | --------------------------------------------------------------- |
| Collane Mobili               | University of Trento                                            |
| TestTeam                     | UCL                                                             |
| Incrementaliser              | Heriot-Watt University                                          |
| SOLO                         | Shanghai Artifiial Intelligence Laboratory, Tsinghua University |
| playornotplay                | Utrecht University                                              |
| bosphorous                   | bKash Limited, Brac Universsity                                 |
| Knock Yourself out           | IIT Roorkee                                                     |
| Bentel rockers               | IIT Bhubaneswar                                                 |
| USTC_NO.1                    | independent researcher                                          |
| BSU-SLIM                     | Boise State University                                          |
| The Architects               | University of Michigan                                          |
| Mediocriphobia               | University of Bonn                                              |
| Maverick                     | University of Stirling                                          |
| Dialogue Architects          | Universität Potsdam                                             |
| NarTeam                      | Narxoz University                                               |
| LLP: Large Language Problems | Potsdam University, Eurac Research                              |
| Playgym                      | Radboud University                                              |



