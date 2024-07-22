from mmengine.config import read_base

with read_base():
    from ..mmlu.mmlu_gen_4d595a import mmlu_datasets
    from ..ceval.ceval_gen_5f30c7 import ceval_datasets
    from ..cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from ..agieval.agieval_gen_64afd3 import agieval_datasets
    from ..GaokaoBench.GaokaoBench_gen_5cfe9e import GaokaoBench_datasets

    from ..gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ..math.math_gen_265cce import math_datasets

    from ..bbh.bbh_gen_5b92b0 import bbh_datasets
    from ..humaneval.humaneval_gen_8e312c import humaneval_datasets
    from ..mbpp.mbpp_gen_1e1056 import mbpp_datasets  # noqa: F401, F403

    #from ..FewCLUE_csl.FewCLUE_csl_gen_28b223 import csl_datasets  # noqa: F401, F403
    #from ..FewCLUE_eprstmt.FewCLUE_eprstmt_gen_740ea0 import eprstmt_datasets  # noqa: F401, F403
    #from ..FewCLUE_chid.FewCLUE_chid_gen_0a29a2 import chid_datasets  # noqa: F401, F403
    # ————————————————————————————————————————————————————————————————————————————————————————————————————


    # from ..ARC_c.ARC_c_gen_1e0de5 import ARC_c_datasets
    # from ..ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    # from ..SuperGLUE_AX_b.SuperGLUE_AX_b_gen_4dfefa import AX_b_datasets
    # from ..SuperGLUE_AX_g.SuperGLUE_AX_g_gen_68aac7 import AX_g_datasets
    # from ..SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_883d50 import BoolQ_datasets
    # from ..SuperGLUE_CB.SuperGLUE_CB_gen_854c6c import CB_datasets
    # from ..SuperGLUE_COPA.SuperGLUE_COPA_gen_91ca53 import COPA_datasets
    # from ..SuperGLUE_MultiRC.SuperGLUE_MultiRC_gen_27071f import MultiRC_datasets
    # from ..SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen_30dea0 import ReCoRD_datasets
    # from ..SuperGLUE_RTE.SuperGLUE_RTE_gen_68aac7 import RTE_datasets
    # from ..SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
    # from ..SuperGLUE_WSC.SuperGLUE_WSC_gen_fe4bf3 import WSC_datasets
    # from ..piqa.piqa_gen_1194eb import piqa_datasets
    # from ..race.race_gen_69ee4f import race_datasets
    # from ..Xsum.Xsum_gen_31397e import Xsum_datasets
    # from ..summedits.summedits_gen_315438 import summedits_datasets
    # from ..obqa.obqa_gen_9069e4 import obqa_datasets
    # from ..lambada.lambada_gen_217e11 import lambada_datasets  # noqa: F401, F403

    # from ..lcsts.lcsts_gen_8ee1fe import lcsts_datasets
    # from ..xiezhi.xiezhi_gen_b86cf5 import xiezhi_datasets
    # from ..hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    # ————————————————————————————————————————————————————————————————————————————————————————————————————

    # from ..siqa.siqa_gen_e78df3 import siqa_datasets
    # from ..CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets  # noqa: F401, F403
    # from ..commonsenseqa.commonsenseqa_gen_c946f2 import commonsenseqa_datasets  # noqa: F401, F403
    # from ..commonsenseqa_cn.commonsenseqacn_gen_d380d0 import commonsenseqacn_datasets
    # from ..triviaqa.triviaqa_gen_2121ce import triviaqa_datasets  # noqa: F401, F403
    # from ..CLUE_afqmc.CLUE_afqmc_gen_901306 import afqmc_datasets  # noqa: F401, F403
    # from ..CLUE_cmnli.CLUE_cmnli_gen_1abf97 import cmnli_datasets  # noqa: F401, F403
    # from ..CLUE_ocnli.CLUE_ocnli_gen_c4cb6c import ocnli_datasets  # noqa: F401, F403

    # from ..flores.flores_gen_806ede import flores_datasets  # noqa: F401, F403




datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
