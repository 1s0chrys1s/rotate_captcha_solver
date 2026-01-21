
import pytest
import os
import json
from rotate.src.double_rotate_solver import double_rotate_identify

# --- Pytest 数据加载与参数化 ---

def load_test_cases():
    """
    从 angle_result.json 加载测试数据并为 pytest 参数化做准备。
    它会过滤掉任何包含 'error' 字段的条目。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'captcha_imgs', 'angle_result.json')

    if not os.path.exists(json_path):
        pytest.fail(f"必需的结果文件未找到: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    test_cases = []
    for identifier, expected in all_results.items():
        # 跳过 JSON 中已经存在的错误记录
        if 'error' in expected:
            continue
        
        # 创建一个自定义测试ID，以便在 pytest 输出中获得更清晰的报告
        test_id = f"图片对_{identifier}"
        test_cases.append(pytest.param(identifier, expected, id=test_id))
    
    if not test_cases:
        pytest.skip("在 angle_result.json 中没有找到有效的测试用例。所有条目都可能包含错误，或者文件为空。")
        
    return test_cases

# --- 测试函数 ---

@pytest.mark.parametrize("identifier, expected_data", load_test_cases())
def test_angle_consistency(identifier, expected_data):
    """
    测试 double_rotate_identify 函数在单个图片对上的表现。
    该测试是参数化的，会为 JSON 文件中的每个有效条目运行。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, 'captcha_imgs')

    inner_image_path = os.path.join(image_dir, expected_data['inner_image'])
    outer_image_path = os.path.join(image_dir, expected_data['outer_image'])
    expected_angle = expected_data['predicted_angle']

    # 在运行测试前，断言图片文件存在
    assert os.path.exists(inner_image_path), f"内圈图片未找到: {inner_image_path}"
    assert os.path.exists(outer_image_path), f"外圈图片未找到: {outer_image_path}"

    # 调用函数以获取实际结果
    actual_match_data = double_rotate_identify(
        small_circle=inner_image_path,
        big_circle=outer_image_path,
        image_type=2,
        speed_ratio=1,
        grayscale=False,
        standard_deviation=0,
        cut_pixel_value=0,
        check_pixel=10,
    )
    
    assert actual_match_data is not None, "函数返回了 None。"
    actual_angle = actual_match_data.total_rotate_angle

    # 误差0.5度以内都算通过
    assert actual_angle == pytest.approx(expected_angle, abs=0.0)
    
    print(f"图片对 {identifier}: OK (预期 ≈{expected_angle:.2f}, 得到 {actual_angle:.2f})")
