# 离线模型资产恢复与软件推理

`build/cyclegan_model`（`params_manifest.json`、`generator_params_q88.mem`、
参考向量）最初只在原开发机上生成。本目录新增两个独立工具，使仓库在没有
这些生成产物的情况下，也能从 RTL 源码本身重建资产并在 PC 上运行整网推理。
不修改任何生产 RTL / 调度脚本，原工具链行为不变。

## tools/recover_model_assets.py

从 RTL ROM 源码恢复 `build/cyclegan_model` 资产：

* 权重：解包 `rtl/mn_generator_rom.sv`（22 个 10-bit 低位面 + 3 个符号位面，
  `rtl/mn_model_params_init.vh` 中的 10 个例外字补丁），
  得到 epoch11 Q8.8 参数 0..21602。
* 结构：`params_manifest.json`。各层参数基址取自
  `rtl/cyclegan_encoder.sv`（编码器 dw/db/pw/pb_base）、
  `docs/cyclegan_stage3_deployment.md`（转置层与输出层起点），
  块内布局（dw/dw_bn/pw/pw_bn = 288/64/1024/64）由
  `tools/prepare_instance_norm_model.py` 的断言值（1067/14027/20427/21603）锁定。
* 测试输入：程序生成的 320x240 合成图（`vectors/frame_rgbx.bin`）。

解包约定（厂商 EFX_RAM10 的 `mem[256*k+i]=INIT_k[i]`、LSB 优先、
符号面按 `addr[14:13]` 分段）由交叉验证锁定：

* `mn_encoder_rom.sv`（直接 16-bit 存储）与生成器 ROM 解包结果的前 1067 字一致；
* `mn_residual_rom.sv` 14027 字、`mn_decoder_rom.sv` 20426 字两份独立副本
  与生成器 ROM 逐字一致；
* 全部非例外字满足符号扩展规则（`((v>>10)&63)==(63 if v&1024 else 0)`）。

```powershell
python tools/recover_model_assets.py   # 约 5 秒，输出 RECOVER_MODEL_ASSETS_PASS
```

依赖：Python 3 + NumPy + Pillow。运行一次即可，产物已生成时可直接跳过。

## tools/run_recovered_model.py

在 PC 上运行与 FPGA 调度等价的整网指令级整数模型
（复用 `tools/cyclegan_host.py` 的 `ArrayCore`/`GeneratorScheduler`，
无 `.npy` 逐层参考比对——参考向量仅存在于原开发机）。

```powershell
# 默认测试图
python tools/run_recovered_model.py

# 任意照片，保持原图比例（竖幅自动旋转进出网络；非 4:3 横幅边缘延展补边）
python tools/run_recovered_model.py --image "build\input_photo.jpg" --out "shg.png"
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--image` | 内置合成图 | 输入照片，任意尺寸，保持原图比例 |
| `--out` | `build/cyclegan_schedule/output.png` | 输出 PNG |
| `--final-scale` | 1.0 | 放大输出层(20427..21602)权重，拉伸 Tanh 输入，增强对比度/饱和度（建议 1.5~3.5） |
| `--bn-scale` | 1.0 | 放大全部 BN scale(a)（偶数偏移），增强逐层风格化（建议 ≤1.25，>1.3 会饱和崩坏） |

单次整网约 12 秒（Python 逐指令模拟）。输出为 Q4.4 定点模型的结果，
与板上实时输出同一数值基准；风格增强参数只改 PC 端副本，不影响 RTL。

## 边界说明

* 恢复的权重已通过四份 ROM 交叉验证与指令计数核对
  （DW 8064 / PW 14208 / NORM 28416 / ADD 3456 / CONV 116736，
  与通道数×tile 数逐项吻合），但没有原机的逐层 `.npy` 参考可做最终比对。
* `params_manifest.json` 中的 `checkpoint_sha256` 沿用
  `docs/analysis/cyclegan_stage1_deployment.json` 的记录值，仅作溯源标识。
* 该定点模型的画质弱于浮点原模型（Q4.4 量化 + 固定归一化），
  与板上部署的已知结论一致。
