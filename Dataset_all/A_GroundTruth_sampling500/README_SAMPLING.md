# 数据采样说明

## 采样方法

本目录包含从原始 `A_GroundTruth` 文件夹中采样的数据。采样方法如下：

1. 对每个领域文件 (issp_answer_xxx.json)，按照受访者的国家代码进行均匀采样
2. 每个领域采样约500条数据
3. 采样逻辑：
   - 首先识别每个记录中的国家代码字段
   - 按国家分组数据
   - 计算每个国家需要采样的数量 (目标总数500 / 国家数量)
   - 从每个国家中随机采样相应数量的记录
   - 如果某个国家的记录数少于需要采样的数量，则使用该国家的所有记录

## 采样结果

| 领域 | 原始记录数 | 采样记录数 | 国家数量 | 每国采样数 |
|------|------------|------------|----------|------------|
| citizenship | 49,807 | 476 | 34 | 14 |
| environment | 44,100 | 476 | 28 | 17 |
| family | 61,754 | 492 | 41 | 12 |
| health | 44,549 | 480 | 30 | 16 |
| nationalidentity | 45,297 | 481 | 37 | 13 |
| religion | 46,267 | 495 | 33 | 15 |
| roleofgovernment | 48,720 | 490 | 35 | 14 |
| socialinequality | 44,975 | 493 | 29 | 17 |
| socialnetworks | 44,492 | 480 | 30 | 16 |
| workorientations | 51,668 | 481 | 37 | 13 |

## 文件说明

- `issp_answer_*.json`: 采样后的答案文件
- `issp_profile_*.json`: 从原目录复制的资料文件
- `README_SAMPLING.md`: 本文件，说明采样方法和结果

## 详细采样信息

### citizenship

| 国家代码 | 采样数量 |
|----------|----------|
| Australia | 14 |
| Austria | 14 |
| Chile | 14 |
| Taiwan | 14 |
| Croatia | 14 |
| Czech Republic | 14 |
| Denmark | 14 |
| Finland | 14 |
| France | 14 |
| Georgia | 14 |
| Hungary | 14 |
| Iceland | 14 |
| India | 14 |
| Japan | 14 |
| Korea (South) | 14 |
| Lithuania | 14 |
| Netherlands | 14 |
| Norway | 14 |
| Philippines | 14 |
| Poland | 14 |
| Russia | 14 |
| Slovakia | 14 |
| Slovenia | 14 |
| South Africa | 14 |
| Spain | 14 |
| Sweden | 14 |
| Switzerland | 14 |
| Turkey | 14 |
| United States of America | 14 |
| Venezuela | 14 |
| Belgium | 14 |
| Germany | 14 |
| Israel | 14 |
| Great Britain | 14 |

### environment

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 17 |
| Australia | 17 |
| Switzerland | 17 |
| China | 17 |
| Germany | 17 |
| Denmark | 17 |
| Spain | 17 |
| Finland | 17 |
| France | 17 |
| Croatia | 17 |
| Hungary | 17 |
| India | 17 |
| Iceland | 17 |
| Italy | 17 |
| Japan | 17 |
| Korea (South) | 17 |
| Lithuania | 17 |
| Norway | 17 |
| New Zealand | 17 |
| Philippines | 17 |
| Russia | 17 |
| Sweden | 17 |
| Slovenia | 17 |
| Slovakia | 17 |
| Thailand | 17 |
| Taiwan | 17 |
| United Stated | 17 |
| South Africa | 17 |

### family

| 国家代码 | 采样数量 |
|----------|----------|
| Argentina | 12 |
| Australia | 12 |
| Austria | 12 |
| Bulgaria | 12 |
| Canada | 12 |
| Chile | 12 |
| China | 12 |
| Taiwan | 12 |
| Croatia | 12 |
| Czech Republic | 12 |
| Denmark | 12 |
| Finland | 12 |
| France | 12 |
| Hungary | 12 |
| Iceland | 12 |
| India | 12 |
| Ireland | 12 |
| Israel | 12 |
| Japan | 12 |
| Korea (South) | 12 |
| Latvia | 12 |
| Lithuania | 12 |
| Mexico | 12 |
| Netherlands | 12 |
| Norway | 12 |
| Philippines | 12 |
| Poland | 12 |
| Russia | 12 |
| Slovakia | 12 |
| Slovenia | 12 |
| South Africa | 12 |
| Spain | 12 |
| Sweden | 12 |
| Switzerland | 12 |
| Turkey | 12 |
| United States of America | 12 |
| Venezuela | 12 |
| Belgium | 12 |
| Germany | 12 |
| Portugal | 12 |
| Great Britain | 12 |

### health

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 16 |
| Australia | 16 |
| Switzerland | 16 |
| China | 16 |
| Czech Republic | 16 |
| Germany | 16 |
| Denmark | 16 |
| Finland | 16 |
| France | 16 |
| Croatia | 16 |
| Hungary | 16 |
| Israel | 16 |
| India | 16 |
| Iceland | 16 |
| Italy | 16 |
| Japan | 16 |
| Mexico | 16 |
| Netherlands | 16 |
| Norway | 16 |
| New Zealand | 16 |
| Philippines | 16 |
| Poland | 16 |
| Russia | 16 |
| Slovenia | 16 |
| Slovakia | 16 |
| Suriname | 16 |
| Thailand | 16 |
| Taiwan | 16 |
| United Stated | 16 |
| South Africa | 16 |

### nationalidentity

| 国家代码 | 采样数量 |
|----------|----------|
| Taiwan | 13 |
| Croatia | 13 |
| Czechia | 13 |
| Denmark | 13 |
| Estonia | 13 |
| Finland | 13 |
| France | 13 |
| Georgia | 13 |
| Hungary | 13 |
| Iceland | 13 |
| India | 13 |
| Ireland | 13 |
| Japan | 13 |
| South Korea | 13 |
| Latvia | 13 |
| Lithuania | 13 |
| Mexico | 13 |
| Norway | 13 |
| Philippines | 13 |
| Russia | 13 |
| Slovakia | 13 |
| Slovenia | 13 |
| South Africa | 13 |
| Spain | 13 |
| Sweden | 13 |
| Switzerland | 13 |
| Turkey | 13 |
| United States | 13 |
| Germany (West) | 13 |
| Germany (East) | 13 |
| Israel – Jews | 13 |
| Israel – Arabs | 13 |
| Belgium–Flanders | 13 |
| Belgium–Wallonia | 13 |
| Belgium–Brussels-Capital Region | 13 |
| Portugal | 13 |
| United Kingdom – Great Britain | 13 |

### religion

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 15 |
| Bulgaria | 15 |
| Switzerland | 15 |
| Chile | 15 |
| Czech Republic | 15 |
| Germany | 15 |
| Denmark | 15 |
| Spain | 15 |
| Finland | 15 |
| France | 15 |
| Great Britain | 15 |
| Georgia | 15 |
| Croatia | 15 |
| Hungary | 15 |
| Israel | 15 |
| Iceland | 15 |
| Italy | 15 |
| Japan | 15 |
| Korea (South) | 15 |
| Lithuania | 15 |
| Norway | 15 |
| New Zealand | 15 |
| Philippines | 15 |
| Russia | 15 |
| Sweden | 15 |
| Slovenia | 15 |
| Slovakia | 15 |
| Suriname | 15 |
| Thailand | 15 |
| Turkey | 15 |
| Taiwan | 15 |
| United Stated | 15 |
| South Africa | 15 |

### roleofgovernment

| 国家代码 | 采样数量 |
|----------|----------|
| Australia | 14 |
| Belgium | 14 |
| Switzerland | 14 |
| Chile | 14 |
| Czech Republic | 14 |
| Germany | 14 |
| Denmark | 14 |
| Spain | 14 |
| Finland | 14 |
| France | 14 |
| Great Britain | 14 |
| Georgia | 14 |
| Croatia | 14 |
| Hungary | 14 |
| Israel | 14 |
| India | 14 |
| Iceland | 14 |
| Japan | 14 |
| Korea (South) | 14 |
| Lithuania | 14 |
| Latvia | 14 |
| Norway | 14 |
| New Zealand | 14 |
| Philippines | 14 |
| Russia | 14 |
| Sweden | 14 |
| Slovenia | 14 |
| Slovakia | 14 |
| Suriname | 14 |
| Thailand | 14 |
| Turkey | 14 |
| Taiwan | 14 |
| United Stated | 14 |
| Venezuela | 14 |
| South Africa | 14 |

### socialinequality

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 17 |
| Australia | 17 |
| Bulgaria | 17 |
| Switzerland | 17 |
| Chile | 17 |
| Czech Republic | 17 |
| Germany | 17 |
| Denmark | 17 |
| Finland | 17 |
| France | 17 |
| Great Britain | 17 |
| Croatia | 17 |
| Israel | 17 |
| Iceland | 17 |
| Italy | 17 |
| Japan | 17 |
| Lithuania | 17 |
| Norway | 17 |
| New Zealand | 17 |
| Philippines | 17 |
| Russia | 17 |
| Sweden | 17 |
| Slovenia | 17 |
| Suriname | 17 |
| Thailand | 17 |
| Taiwan | 17 |
| United Stated | 17 |
| Venezuela | 17 |
| South Africa | 17 |

### socialnetworks

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 16 |
| Australia | 16 |
| Switzerland | 16 |
| China | 16 |
| Czech Republic | 16 |
| Germany | 16 |
| Denmark | 16 |
| Spain | 16 |
| Finland | 16 |
| France | 16 |
| Great Britain | 16 |
| Croatia | 16 |
| Hungary | 16 |
| Israel | 16 |
| India | 16 |
| Iceland | 16 |
| Japan | 16 |
| Lithuania | 16 |
| Mexico | 16 |
| New Zealand | 16 |
| Philippines | 16 |
| Russia | 16 |
| Sweden | 16 |
| Slovenia | 16 |
| Slovakia | 16 |
| Suriname | 16 |
| Thailand | 16 |
| Taiwan | 16 |
| United Stated | 16 |
| South Africa | 16 |

### workorientations

| 国家代码 | 采样数量 |
|----------|----------|
| Austria | 13 |
| Australia | 13 |
| Belgium | 13 |
| Switzerland | 13 |
| Chile | 13 |
| China | 13 |
| Czech Republic | 13 |
| Germany | 13 |
| Denmark | 13 |
| Estonia | 13 |
| Spain | 13 |
| Finland | 13 |
| France | 13 |
| Great Britain | 13 |
| Georgia | 13 |
| Croatia | 13 |
| Hungary | 13 |
| Israel | 13 |
| India | 13 |
| Iceland | 13 |
| Japan | 13 |
| Lithuania | 13 |
| Latvia | 13 |
| Mexico | 13 |
| Norway | 13 |
| New Zealand | 13 |
| Philippines | 13 |
| Poland | 13 |
| Russia | 13 |
| Sweden | 13 |
| Slovenia | 13 |
| Slovakia | 13 |
| Suriname | 13 |
| Taiwan | 13 |
| United Stated | 13 |
| Venezuela | 13 |
| South Africa | 13 |

