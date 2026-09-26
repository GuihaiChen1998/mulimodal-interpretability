# COCO → Steerling 概念映射 v1（合并版，待用户审阅）

> 2026-09-26。文件：`results/week1/e7/coco2concept_v1_reviewed.json`（每类保留前 6 个候选及各项分数）；自动合并前的版本是 `coco2concept_v1.json`。
> 脚本：`scripts/e7c_merge_coco_map.py`（合并），`scripts/e7d_review_coco_map.py`（Claude 的判定和修改）。

## 合并规则
每个类别的候选池 = 按 Steerling 读出特异性取前 10 ∪ 按 CLIP 文本相似度取前 10 ∪ E7a 的前 10 个候选，用三种信号打分：

| 信号 | 含义 |
|---|---|
| spec | 在描述里把类别词 mask 掉，读出该位置的概念；spec = 激活率 × IDF（IDF 按 15K 条描述计算）。每类最多取 80 条 COCO 描述。对描述里少见的类别名加了同义词，例如 handbag→purse、hair drier→hair dryer、sports ball→soccer ball/tennis ball/…、tv→television、couch→sofa |
| clip | CLIP 文本相似度："a photo of a {类别}" 与 "a photo of {概念名}" |
| lex | 概念名称或组名里含该词（name_hit）；概念嵌入经 LM head 会促进该词（token_hit） |

- 总分 = 0.5·spec（池内归一化）+ 0.3·clip（池内归一化）+ 0.1·name_hit + 0.1·token_hit。
- 自动置信度为 high 的条件：主概念同时是 spec 第一和 clip 第一，或领先第二名 ≥ 0.15。自动结果中 high 为 47/80。
- Claude 在此基础上逐类判定，并修改了 10 类的主概念，见下表"修改"一列。

## 结论
- 判定结果：**✅ 精确 37 类 / 🟡 相关或上位 42 类 / ❌ 无合适概念 1 类**（bench）。
- 80 个类别只对应到 **62 个不同的概念**，有 4 组类别共用同一个概念：
  - backpack / handbag / suitcase → Bags and Storage Containers
  - skis / snowboard / skateboard → Skating and Sk- Sports
  - fork / knife / spoon → Knives and Blades
  - microwave / oven / toaster → Microwave ovens and heating

  这说明 Steerling 的概念粒度比 COCO 类别粗。第 2 周算"概念–物体 AUROC"时，应该按**概念**合并这些类别来评测，不能按 80 类评测。
- **名称最贴切的概念，往往在 mask 位置上从不激活**，例如 "Kite Flying"、"Pizza and Italian Cuisine"、"Toilets and Sanitation"、"Typing and Keyboards"，spec = 0。模型实际用的是另一个近义概念（很多概念名称重复出现）。
  - 对 M2 来说，应优先选模型真正会用的概念，所以合并时 spec 的权重最高。
  - 唯一例外是 kite：自动结果选了 "Drones and UAVs"，Claude 改成了从不激活的 "Kite Flying"，需要你决定保留哪一个。
- 相关或上位概念主要集中在三类：
  - 餐具和器皿（cup→Plates and tableware，fork 和 spoon→Knives and Blades）；
  - 运动器材（skis、surfboard、baseball bat 都只对到运动项目）；
  - 路边设施（fire hydrant→Firefighting，stop sign→Signs，parking meter→Parking）。

  这些物体在 33K 概念里没有物体级的条目，属于概念覆盖的局限，论文里应如实说明。

## 完整表格
| COCO | 主概念 (id) | 判定 | 自动置信度 | 描述数 | 修改 |
|---|---|---|---|---|---|
| person | Humans and Living Beings (31442) | ✅ 精确 | review | 80 | 原为 27293：top spec; object-level 'humans' beats 'Specific Individuals' |
| bicycle | Bicycles and Motorcycles (5125) | ✅ 精确 | high | 80 |  |
| car | Cars and vehicles (21762) | ✅ 精确 | high | 80 |  |
| motorcycle | Bicycles and Motorcycles (5125) | ✅ 精确 | high | 80 |  |
| airplane | Aircraft and Aerospace Engineering (10605) | ✅ 精确 | review | 80 | 原为 29600：higher spec and clip; 'Planes' also covers geometry |
| bus | Bus (vehicle and software) (2628) | ✅ 精确 | high | 80 |  |
| train | Trains and Railways (32179) | ✅ 精确 | high | 80 |  |
| truck | Trucks, Trailers, and RVs (29519) | ✅ 精确 | high | 80 |  |
| boat | Ships and Maritime Activity (32110) | ✅ 精确 | high | 80 |  |
| traffic light | Traffic Signals and Intersections (3643) | ✅ 精确 | review | 80 |  |
| fire hydrant | Firefighting and Fire Safety (31326) | 🟡 相关/上位 | review | 80 |  |
| stop sign | Signs, signals, signatures (33316) | 🟡 相关/上位 | review | 80 |  |
| parking meter | Parking and Driving Directions (4507) | 🟡 相关/上位 | high | 80 |  |
| bench | Architectural Building Elements (4492) | ❌ 无合适概念 | review | 80 |  |
| bird | Birds and Ornithology (31730) | ✅ 精确 | high | 80 |  |
| cat | Cats and feline references (14969) | ✅ 精确 | high | 80 |  |
| dog | Dog laws and regulations (28636) | ✅ 精确 | high | 80 |  |
| horse | Horses and Equestrianism (19778) | ✅ 精确 | high | 80 |  |
| sheep | Sheep, Wool, and Shearing (5935) | ✅ 精确 | high | 80 |  |
| cow | Cattle and Dairy Husbandry (32876) | ✅ 精确 | review | 80 |  |
| elephant | Elephants, Ivory, and Megafauna Conservation (18777) | ✅ 精确 | high | 80 |  |
| bear | North American Wildlife and Hunting (29751) | 🟡 相关/上位 | high | 80 |  |
| zebra | African Wildlife and Safari (26481) | 🟡 相关/上位 | review | 80 | 原为 24265：higher spec; same concept as giraffe |
| giraffe | African Wildlife and Safari (26481) | 🟡 相关/上位 | review | 80 |  |
| backpack | Bags and Storage Containers (28829) | 🟡 相关/上位 | high | 19 |  |
| umbrella | Rain and Precipitation (28218) | 🟡 相关/上位 | high | 80 |  |
| handbag | Bags and Storage Containers (28829) | 🟡 相关/上位 | high | 12 |  |
| tie | Dress codes and etiquette (27557) | 🟡 相关/上位 | review | 80 |  |
| suitcase | Bags and Storage Containers (28829) | 🟡 相关/上位 | high | 80 |  |
| frisbee | Throwing and ball sports (32002) | 🟡 相关/上位 | review | 80 |  |
| skis | Skating and Sk- Sports (29412) | 🟡 相关/上位 | review | 80 |  |
| snowboard | Skating and Sk- Sports (29412) | 🟡 相关/上位 | review | 66 |  |
| sports ball | Throwing and ball sports (32002) | ✅ 精确 | review | 80 |  |
| kite | Kite Flying (21555) | ✅ 精确 | review | 80 | 原为 6719：exact 'Kite Flying' (clip 0.91) though inactive in masked read-out; current pick is 'Drones' |
| baseball bat | Baseball Gameplay and MLB (33045) | 🟡 相关/上位 | high | 80 |  |
| baseball glove | Baseball Gameplay and MLB (33045) | 🟡 相关/上位 | high | 30 |  |
| skateboard | Skating and Sk- Sports (29412) | 🟡 相关/上位 | review | 80 | 原为 5125：current pick is 'Bicycles and Motorcycles'; highest clip in pool |
| surfboard | Sailing and Board Sports (990) | 🟡 相关/上位 | review | 80 |  |
| tennis racket | Racket sports and tennis (12743) | ✅ 精确 | high | 80 |  |
| bottle | Beverages and Alcoholic Drinks (13021) | 🟡 相关/上位 | high | 80 |  |
| wine glass | Beverages and Alcoholic Drinks (13021) | 🟡 相关/上位 | high | 37 |  |
| cup | Plates and tableware (17856) | 🟡 相关/上位 | review | 80 | 原为 14987：current pick is the measuring-cup sense |
| fork | Knives and Blades (19825) | 🟡 相关/上位 | high | 80 |  |
| knife | Knives and Blades (19825) | ✅ 精确 | high | 66 |  |
| spoon | Knives and Blades (19825) | 🟡 相关/上位 | high | 52 |  |
| bowl | Plates and tableware (17856) | ✅ 精确 | high | 80 |  |
| banana | Fruits and ripening (23049) | 🟡 相关/上位 | high | 80 |  |
| apple | Apple (fruit and Inc.) (18455) | ✅ 精确 | high | 80 |  |
| sandwich | Sandwiches and Condiments (29115) | ✅ 精确 | high | 80 |  |
| orange | Fruits and ripening (23049) | 🟡 相关/上位 | review | 80 |  |
| broccoli | Vegetables and produce preparation (17182) | 🟡 相关/上位 | review | 80 | 原为 26510：vegetable-specific; consistent with carrot |
| carrot | Vegetables and produce preparation (17182) | 🟡 相关/上位 | review | 80 |  |
| hot dog | Fast Food Chains and Burgers (7322) | 🟡 相关/上位 | review | 80 |  |
| pizza | Pizza and Countertop Ovens (13097) | ✅ 精确 | high | 80 |  |
| donut | Sweets and Desserts (17504) | 🟡 相关/上位 | review | 80 |  |
| cake | Sweets and Desserts (17504) | 🟡 相关/上位 | high | 80 | 原为 22045：clip 0.82 vs 0.73; 'Holiday Feast Foods' is too broad |
| chair | Ergonomic seating and workspace comfort (8147) | ✅ 精确 | high | 80 |  |
| couch | Moving and rearranging furniture (7136) | 🟡 相关/上位 | review | 80 |  |
| potted plant | Botany and Plant Science (21313) | 🟡 相关/上位 | high | 80 |  |
| bed | Beds and Bedding (16092) | ✅ 精确 | high | 80 |  |
| dining table | Tables and Furniture (29785) | ✅ 精确 | high | 80 |  |
| toilet | Bathrooms and Bathing Fixtures (32093) | 🟡 相关/上位 | review | 80 |  |
| tv | Television and Broadcasting (9704) | ✅ 精确 | high | 80 |  |
| laptop | Personal Computer Hardware and OS (27802) | ✅ 精确 | high | 80 |  |
| mouse | Computer mouse and pointers (26261) | ✅ 精确 | review | 80 |  |
| remote | Gaming Hardware and Controllers (24949) | 🟡 相关/上位 | review | 80 |  |
| keyboard | Keyboard Input Handling (32211) | ✅ 精确 | high | 80 |  |
| cell phone | Phones and Personal Devices (27933) | ✅ 精确 | high | 80 |  |
| microwave | Microwave ovens and heating (15005) | ✅ 精确 | high | 69 |  |
| oven | Microwave ovens and heating (15005) | 🟡 相关/上位 | high | 80 |  |
| toaster | Microwave ovens and heating (15005) | 🟡 相关/上位 | high | 17 |  |
| sink | Bathrooms and Bathing Fixtures (32093) | 🟡 相关/上位 | high | 80 |  |
| refrigerator | Home Appliance Repair (33499) | 🟡 相关/上位 | review | 80 | 原为 33346：current pick is 'Cool' slang; tie on score |
| book | Books and publishing (20353) | ✅ 精确 | high | 80 |  |
| clock | Watches and Timekeeping (29888) | ✅ 精确 | high | 80 |  |
| vase | Ancient Greek Pottery Vessels (15958) | 🟡 相关/上位 | review | 80 |  |
| scissors | Sharp Blades and Edges (17184) | 🟡 相关/上位 | review | 62 |  |
| teddy bear | Children's toys and play (6832) | 🟡 相关/上位 | review | 80 | 原为 33610：toy concept instead of 'Cute Affectionate Child Scenes' |
| hair drier | Hair and Hairstyling (20299) | 🟡 相关/上位 | review | 14 |  |
| toothbrush | Dental hygiene products (14795) | ✅ 精确 | review | 50 |  |
