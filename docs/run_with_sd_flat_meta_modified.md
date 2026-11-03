*#  Speculative Decoding mode*

    *# Prefill: Target 和 Draft 都对相同 token 做 forward, 同时建立两份 KV; 只有 Target 决定最终输出 token*

    *# Decode: Draft 自回归产生最多 K 个 proposal; Target 一次 ragged forward 并行验证; 返回 accepted proposals + correction/bonus*

    *# 返回：{seq_id: [token1, token2, ...]};对 decode 而言长度最多 K + 1，最后一个 token 一定是Target 给出的 correction 或 bonus*

    **@torch.no_grad**()

    def **run_with_sd**(self, output: SchedulerOutput) -> dict[int, list[int]]:

        """

        假设当前 SchedulerOutput 中包含：

        x 个 prefill item

        y 个 decode item



        整体分为三个阶段：

        Step 1.1: 所有 prefill item 一次性送入 Draft Model, 建立对应的 Draft KV Cache

        Step 1.2: 所有 decode item 以 batch 形式进行 K 轮 Draft decode, 得到每个请求的 d1...dK 及对应 q(x); 最后额外把 dK 再 forward 一次(写入kv cache)

        Step 2: 将 prefill chunk + decode verification segment [pending, d1, ..., dK] 重新组成一个 mixed batch, 一次性送入 Target Model;

                对 decode verification segment 的每一个位置都保留 logits, 从而得到验证 proposal 所需的 p(x) 和最后的 bonus distribution

        Step 3: 对每个 decode item 执行 accept / reject; 返回 accepted draft tokens + correction/bonus

        Return: dict[int, list[int]]



        例如：

            {

                seq0: [first_output],                # final prefill

                seq1: [d1, d2, correction],          # SD reject

                seq2: [d1, d2, d3, d4, bonus],       # all accepted

            }

        """



        if not output.items:

            return {}

        if self.draft_model is None or self.draft_kv_cache is None:

            raise RuntimeError("Fatal: Speculative decoding requires draft model and draft KV cache.")



        results: dict[int, list[int]] = {}

        prefill_items = [item for item in output.items if item.is_prefill]

        decode_items = [item for item in output.items if not item.is_prefill]



        *# 确定本批batch中decode item共同的最大可推理长度*

        *# 当前实现每个seq的K都是固定的, 等于_max_draft_tokens; 且受到每个seq各自剩余token len(context/outut)的约束*

        K = self.\_max_draft_tokens

        for item in decode_items:

            seq = item.seq

            remaining_output = (seq.sampling_params.max_tokens - seq.num_completion_tokens)

            remaining_context = (self.config.max_model_len - seq.num_tokens)

            K = **min**(K, **max**(0, remaining_output - 1), **max**(0, remaining_context - 1)) *# 受到三者的共同约束*



        *# 为每个 decode seq 的 draft proposal 和 q(x) 预留空间, 并记录每个 decode 起点*

        draft_tokens: dict[int, list[int]] = {item.seq.seq_id: [] for item in decode_items}

        draft_probs: dict[int, list[torch.Tensor]] = {item.seq.seq_id: [] for item in decode_items}

        decode_start: dict[int, int] = {item.seq.seq_id: item.start_pos for item in decode_items}



        *# 为保险额外检查draft/target kv cache reservation情况; 相应逻辑理应由schedule处理, 这里是防御性编程*

        *# ModelRunner 只负责 forward，不负责 block allocation; Scheduler / BlockManager 必须提前 reserve 到 start + K*

        for item in decode_items:

            seq = item.seq

            last_write_position = item.start_pos + K

            logical_block = (last_write_position // seq.block_size)



            if logical_block >= **len**(seq.block_table):

                raise RuntimeError("Fatal: Insufficient KV cache reservation for speculative decoding.")





        *# Step 1.1: 将所有 prefill item 打成一个 batch 送入 Draft Model 进行一轮forward*

        if prefill_items:

            prefill_output = SchedulerOutput(prefill_items)

            try:

                input_ids, positions = self.prepare_batch(prefill_output)

                self.draft_model(input_ids, positions, output_hidden_states=False)



                *# Draft KV 已经真实写到各 prefill item 的 end*

                for item in prefill_items:

                    end = item.start_pos + item.num_tokens

                    item.seq.draft_kv_computed_tokens = end



            finally:

                reset_context()





        *# Step 1.2: 所有 decode item 进行 K + 1 轮decode, 前 K 轮采样 + 记录logits向量q(x); 最后1轮不采样*

        if decode_items:

            max_blocks = **max**(**len**(item.seq.block_table) for item in decode_items)



            *# 一共执行 K + 1 次 Draft forwar, 其中前 K 次生成 proposal，最后一次只写KV cache*

            for depth in **range**(K + 1):

                input_token_ids: list[int] = []

                positions: list[int] = []

                slot_mapping: list[int] = []

                context_lens: list[int] = []

                block_tables: list[list[int]] = []



                for item in decode_items: *# 打包所有的decode请求*

                    seq = item.seq

                    seq_id = seq.seq_id

                    start = decode_start[seq_id]    *# 上一轮 pending 的位置*

                    position = start + depth        *# 本轮KV cache 写入的frontier position*

                    token = seq.token_ids[start] if depth == 0 else draft_tokens[seq_id][depth - 1]



                    input_token_ids.append(token)   *# 每个seq请求本轮需要输入模型的token_id*

                    positions.append(position)      *# 每个token的绝对位置*

                    slot_mapping.append(token_to_slot(seq, position)) *# 输入模型的token_id对应的物理kv cache索引, 用于kv cache写入*

                    context_lens.append(position + 1) *# 每个seq请求已经生成的token长度, 用于attention计算时gather历史kv cache*

                    table = list(seq.block_table)    

                    table.extend([-1] \* (max_blocks - **len**(table)))

                    block_tables.append(table)      *# 已分配block id列表, -1用于尾部对齐、代表尚未分配的block*



                batch_size = **len**(decode_items)



                try: *# 对y个decode请求打包, set_context然后进行forward*

                    set_context(

                        q_lens=torch.ones(batch_size, dtype=torch.int32, device=self.device), *# y个1*

                        context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),

                        cu_seqlens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=self.device),

                        max_seqlen_q=1,

                        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),

                        block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),

                        is_prefill=torch.zeros(batch_size, dtype=torch.bool, device=self.device), *# 所有seq都不是prefill*

                        is_sample=torch.full((batch_size,), depth < K, dtype=torch.bool, device=self.device), *# 仅前 K 轮需要采样*

                    )



                    *# 进行forward*

                    input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)

                    positions_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)

                    hidden_states, \_ = self.draft_model(input_ids, positions_tensor, output_hidden_states=False)





                    for item in decode_items: *# 当前输入 token 已经真正写入 Draft KV; 这里遵循右开区间设定, 位置指向下一个未被写入的pos*

                        item.seq.draft_kv_computed_tokens = (decode_start[item.seq.seq_id] + depth + 1)



                    *# 只有前 K 次需要采样*

                    if depth == K:

                        continue

                    *# 准备采样参数*

                    logits = self.draft_model.compute_logits(hidden_states) *# [y, vocab_size]*

                    temperatures = torch.tensor(

                        [item.seq.sampling_params.temperature for item in decode_items],

                        dtype=logits.dtype,

                        device=logits.device,

                    )

                    topks = torch.tensor(  *# shape = [sampled_seq]*

                        [item.seq.sampling_params.topk for item in sample_items],

                        dtype=torch.long,

                        device=logits.device,

                    )

                    greedy_mask = torch.tensor(

                        [item.seq.sampling_params.greedy for item in decode_items],

                        dtype=torch.bool,

                        device=logits.device,

                    )



                    tokens = self.sampler(logits=logits, temperatures=temperatures, topks=topks, greedy_mask=greedy_mask) *# [y]*



                    for row, (item, token) in **enumerate**(**zip**(decode_items, tokens.tolist())):

                        seq = item.seq

                        seq_id = seq.seq_id

                        proposal = int(token)

                        draft_tokens[seq_id].append(proposal)



                        if seq.sampling_params.greedy: *# Greedy SD 不需要概率比 p/q*

                            continue

                        q = distribution_from_logits( *# q(x) 必须来自实际用于 Draft sampling的同一套 temperature / top-k 分布*

                            logits[row],

                            temperature=(seq.sampling_params.temperature),

                        )

                        draft_probs[seq_id].append(q)



                finally: *# 每轮batched decoding结束都要复位context*

                    reset_context()





        *# Step 2: 将step1输出的decode序列和原始的prefill item再拼成一个mixed batch, 送入target model进行一次forward*

        input_ids_flat: list[int] = []

        positions_flat: list[int] = []

        slot_mapping_flat: list[int] = []

        q_lens: list[int] = []

        context_lens: list[int] = []

        is_prefill: list[bool] = []

        is_sample: list[bool] = []

        block_tables: list[list[int]] = []      *# [num_seq]*

        flat_sample_mask: list[tuple[bool, int, int]] = []       *# [num_flat_tokens]*  # [MOD]



        # [MOD] sample_metadata 删除；flat_sample_mask 直接保存 (need_logits, item_idx, inner_idx)



        max_blocks = **max**(**len**(item.seq.block_table) for item in output.items)





        *# Step 2.1: 按 SchedulerOutput 原始 item 顺序重新构造 mixed batch*

        for item_idx, item in **enumerate**(output.items):  # [MOD]

            seq = item.seq

            seq_id = seq.seq_id



            if item.is_prefill: *# 处理原始 prefill item*

                start = item.start_pos

                end = start + item.num_tokens

                tokens = seq.token_ids[start:end]

                positions = list(**range**(start, end))

                input_ids_flat.extend(tokens)       *# 注意这里是直接flatten之后跨seq请求拼接*

                positions_flat.extend(positions)

                slot_mapping_flat.extend(token_to_slot(seq, position) for position in positions)



                q_lens.append(item.num_tokens)      *# 这里是每个seq请求对应一个参数*

                context_lens.append(end)

                is_prefill.append(True)

                is_sample.append(item.sample_after)

                table = list(seq.block_table)

                table.extend([-1] \* (max_blocks - **len**(table)))

                lock_tables.append(table)

                flat_sample_mask.extend((False, item_idx, local_idx) for local_idx in **range**(item.num_tokens)) *# 先默认这个prefill为partial prefill, 全部位置都不采样*  # [MOD]

                if item.sample_after: *# 如果是final prefill再把最后一个mask改为true*

                    flat_sample_mask[-1] = (True, item_idx, item.num_tokens - 1)  # [MOD]





            else: *# 处理 draft decode item*

                start = decode_start[seq_id]

                verify_tokens = [seq.token_ids[start], \*draft_tokens[seq_id]] *# [pending_token, draft decode token]*

                verify_len = **len**(verify_tokens)



                positions = list(**range**(start, start + verify_len))

                input_ids_flat.extend(verify_tokens)    *# input_id、position、slot_mapping均为展平后的向量(token-level metadata), 长度等于num_total_flat_tokens*

                positions_flat.extend(positions)

                slot_mapping_flat.extend(token_to_slot(seq, position) for position in positions)



                q_lens.append(verify_len)               *# 这几个为seq-level metadata, 长度为y*

                context_lens.append(start + verify_len)

                is_prefill.append(False)

                is_sample.append(True)

                table = list(seq.block_table)

                table.extend([-1] \* (max_blocks - **len**(table)))

                block_tables.append(table)



                flat_sample_mask.extend((True, item_idx, local_idx) for local_idx in **range**(verify_len)) *# 对verification segment的每一个token都必须采样*  # [MOD]





        *# q_lens -> cu_seqlens_q*

        cu_q = [0]

        for q_len in q_lens:

            cu_q.append(cu_q[-1] + q_len)



        try:

            set_context( *# 准备context(metadata)*

                q_lens=torch.tensor(q_lens, dtype=torch.int32, device=self.device),

                context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),

                cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=self.device),

                max_seqlen_q=**max**(q_lens, 0),

                slot_mapping=torch.tensor(slot_mapping_flat, dtype=torch.long, device=self.device),

                block_tables=torch.tensor(block_tables, dtype=torch.int32, device=self.device),

                is_prefill=torch.tensor(is_prefill, dtype=torch.bool, device=self.device),

                is_sample=torch.tensor(is_sample, dtype=torch.bool, device=self.device),

            )



            *# 准备input数据*

            input_ids = torch.tensor(input_ids_flat, dtype=torch.long, device=self.device)

            positions = torch.tensor(positions_flat, dtype=torch.long, device=self.device)



            *# 核心: 所有 prefill + 所有 decode verify 只执行一次 Target forward。*

            target_hidden, \_ = self.model(input_ids, positions, output_hidden_states=False)



            *# Target KV 已经按照各自 query segment 正常写入。*

            for item, end in **zip**(output.items, context_lens):

                item.seq.target_kv_computed_tokens = end



            sample_indices = [idx for idx, (flag, _, _) in **enumerate**(flat_sample_mask) if flag]  # [MOD]

            sample_indices_tensor = torch.tensor(sample_indices, dtype=torch.long, device=self.device)

            sample_hidden = target_hidden.index_select(0, sample_indices_tensor)

            target_logits = self.model.compute_logits(sample_hidden)



        finally:

            reset_context()





        # [MOD] 不再将 Target logits 拆回 prefill/decode dict；Step 3 直接按 flat_sample_mask 顺序消费 target_logits



    *# ================================================================*

    *# Step 3*

    *#*

    *# 按 flat_sample_mask 顺序处理所有需要 logits 的位置；item_idx 定位原始 item，inner_idx 定位 item 内部位置。*  # [MOD]

    *#*

    *# proposal:*

    *#*

    *#     [d1, d2, ..., dK]*

    *#*

    *# Target logits:*

    *#*

    *#     p1, p2, ..., pK, p_bonus*

    *#*

    *# 最终返回：*

    *#*

    *# reject:*

    *#*

    *#     [accepted proposals..., correction]*

    *#*

    *# all accepted:*

    *#*

    *#     [d1, ..., dK, bonus]*

    *#*

    *# 因此前 K 个位置最多保留 K 个 Draft token；*

    *# list 最后一个元素始终为 correction / bonus。*

    *# ================================================================*



    batch_proposed = 0

    batch_accepted = 0

    batch_rounds = 0



    final_tokens_map: dict[int, list[int]] = {item.seq.seq_id: [] for item in decode_items}  # [MOD]

    accepted_count_map: dict[int, int] = {item.seq.seq_id: 0 for item in decode_items}  # [MOD]

    rejected_map: dict[int, bool] = {item.seq.seq_id: False for item in decode_items}  # [MOD]



    for item in decode_items:  # [MOD]

        seq_id = item.seq.seq_id

        if **len**(draft_tokens[seq_id]) != K:

            raise RuntimeError(

                "Fatal: Draft proposal count mismatch."

            )



    logit_row = 0  # [MOD]



    for need_logits, item_idx, inner_idx in flat_sample_mask:  # [MOD]

        if not need_logits:

            continue



        item = output.items[item_idx]  # [MOD]

        seq = item.seq

        seq_id = seq.seq_id

        p_logits = target_logits[logit_row]  # [MOD]

        logit_row += 1  # [MOD]



        if item.is_prefill:  # [MOD]

            temperatures = torch.tensor(

                [seq.sampling_params.temperature],

                dtype=p_logits.dtype,

                device=p_logits.device,

            )

            topks = torch.tensor(

                [seq.sampling_params.topk],

                dtype=torch.long,

                device=p_logits.device,

            )

            greedy_mask = torch.tensor(

                [seq.sampling_params.greedy],

                dtype=torch.bool,

                device=p_logits.device,

            )

            token = self.sampler(

                logits=p_logits.unsqueeze(0),

                temperatures=temperatures,

                topks=topks,

                greedy_mask=greedy_mask,

            )

            results[seq_id] = [int(token.item())]

            continue



        if rejected_map[seq_id]:  # [MOD]

            continue



        proposals = draft_tokens[seq_id]  # [MOD]

        final_tokens = final_tokens_map[seq_id]  # [MOD]



        *# ------------------------------------------------------------*

        *# Step 3.1：依次验证 d1...dK。*

        *# ------------------------------------------------------------*



        if inner_idx < K:  # [MOD]

            proposal = proposals[inner_idx]  # [MOD]



            *# ========================================================*

            *# Greedy SD*

            *# ========================================================*



            if seq.sampling_params.greedy:

                target_token = int(

                    torch.argmax(

                        p_logits

                    ).item()

                )



                if target_token == proposal:

                    final_tokens.append(

                        proposal

                    )



                    accepted_count_map[seq_id] += 1  # [MOD]



                    continue



                *# 第一个 mismatch：*

                *# 直接使用 Target argmax 作为 correction。*

                final_tokens.append(

                    target_token

                )



                rejected_map[seq_id] = True  # [MOD]



                continue  # [MOD]



            *# ========================================================*

            *# Sampling SD*

            *# ========================================================*



            p = distribution_from_logits(

                p_logits,

                temperature=(

                    seq.sampling_params.temperature

                ),

            )



            q = draft_probs[seq_id][inner_idx]  # [MOD]



            p_x = p[proposal]

            q_x = q[proposal]



            if q_x.item() <= 0.0:

                raise RuntimeError(

                    "Fatal: sampled proposal has zero Draft probability."

                )



            accept_prob = torch.clamp(

                p_x / q_x,

                max=1.0,

            )



            random_value = torch.rand(

                (),

                device=self.device,

            )



            if random_value < accept_prob:

                final_tokens.append(

                    proposal

                )



                accepted_count_map[seq_id] += 1  # [MOD]



                continue



            *# --------------------------------------------------------*

            *# Draft token 被拒绝：*

            *#*

            *# correction \~ normalize(max(p - q, 0))*

            *# --------------------------------------------------------*



            correction = sample_residual_correction(

                p,

                q,

            )



            final_tokens.append(

                int(correction)

            )



            rejected_map[seq_id] = True  # [MOD]



            continue  # [MOD]



        *# ------------------------------------------------------------*

        *# Step 3.2：*

        *#*

        *# K 个 Draft proposal 全部接受。*

        *#*

        *# 最后一行 Target logits：*

        *#*

        *#     logits_after_dK*

        *#*

        *# 用于采样 bonus token。*

        *# ------------------------------------------------------------*



        if not rejected_map[seq_id]:  # [MOD]

            bonus_logits = p_logits.unsqueeze(0)  # [MOD]



            temperatures = torch.tensor(

                [

                    seq.sampling_params.temperature

                ],

                dtype=bonus_logits.dtype,

                device=bonus_logits.device,

            )



            greedy_mask = torch.tensor(

                [

                    seq.sampling_params.greedy

                ],

                dtype=torch.bool,

                device=bonus_logits.device,

            )



            bonus_token = self.sampler(

                bonus_logits,

                temperatures,

                greedy_mask,

            )



            final_tokens.append(

                int(bonus_token.item())

            )



    *# ------------------------------------------------------------*

    *# Step 3.3：*

    *#*

    *# ModelRunner 这里只负责“决定应该追加哪些 token”。*

    *#*

    *# KV rollback 不在这里执行。*

    *#*

    *# Scheduler postprocess 根据：*

    *#*

    *#     accepted_count = len(final_tokens) - 1*

    *#*

    *# 得到：*

    *#*

    *#     keep_tokens = start + 1 + accepted_count*

    *#*

    *# 再分别把 Target / Draft 的无效 speculative KV tail*

    *# rollback 到 keep_tokens。*

    *# ------------------------------------------------------------*



    for item in decode_items:  # [MOD]

        seq_id = item.seq.seq_id

        results[seq_id] = final_tokens_map[seq_id]  # [MOD]



        if K > 0:

            batch_proposed += K

            batch_accepted += accepted_count_map[seq_id]  # [MOD]

            batch_rounds += 1



    *# ================================================================*

    *# Step 3.4：更新全局 SD statistics。*

    *# ================================================================*



    self.\_drafted_tokens += batch_proposed

    self.\_accepted_tokens += batch_accepted

    self.\_verification_rounds += batch_rounds



    return results
