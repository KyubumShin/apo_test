# apo_ko_setup.py
import shutil
from pathlib import Path
import agentlightning.algorithm.apo as apo_mod

from prompt_template import TEXT_GRADIENT_ANALYSIS_GUIDELINES


def patch_apo_for_korean():
    """APO 라이브러리의 영어 프롬프트를 한국어 프롬프트로 교체"""
    prompts_dir = Path(__file__).parent / "prompts"
    apo_base_dir = Path(apo_mod.__file__).parent
    apo_prompts_dir = apo_base_dir / "prompts"
    
    files = {
        "text_gradient_ko.poml": "text_gradient_variant01.poml",
        "apply_edit_ko.poml": "apply_edit_variant01.poml",
    }
    
    if not apo_prompts_dir.exists():
        print(f"APO 프롬프트 디렉터리를 찾을 수 없습니다: {apo_prompts_dir}")
        return
    
    for ko_file, apo_file in files.items():
        ko_path = prompts_dir / ko_file
        apo_path = apo_prompts_dir / apo_file
        
        if ko_path.exists():
            # text_gradient_ko.poml의 경우 분석 지침을 prompt_template.py에서 가져와 적용
            if ko_file == "text_gradient_ko.poml":
                content = ko_path.read_text(encoding="utf-8")
                content = _apply_analysis_guidelines(content)
                apo_path.write_text(content, encoding="utf-8")
            else:
                shutil.copy(ko_path, apo_path)
        else:
            print(f"{ko_file} 없음")


def _apply_analysis_guidelines(content: str) -> str:
    """text_gradient_ko.poml 내용에 분석 지침을 prompt_template.py에서 적용"""
    # 플레이스홀더를 실제 분석 지침으로 교체
    return content.replace("{{ANALYSIS_GUIDELINES_PLACEHOLDER}}", TEXT_GRADIENT_ANALYSIS_GUIDELINES)


try:
    patch_apo_for_korean()
except Exception as e:
    print(f"APO 패치 실패: {e}")