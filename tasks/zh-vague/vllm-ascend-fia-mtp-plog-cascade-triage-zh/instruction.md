我们在 8 卡昇腾上用 vllm-ascend 开 TP=8 和 MTP 投机解码跑推理时突然挂死，采集的 8 卡日志目录在 logs/plog_tp8_crash/，终端刷屏全是 Rank 0/1/2 的 HCCL EI0006 等待超时，帮我查下到底是哪块 vllm_ascend 代码引起的并把问题修好。
