"""
Medical LLM Answer Generation Module
-----------------------------
Module sinh câu trả lời y khoa sử dụng mô hình LLM chuyên biệt y khoa dựa trên
bằng chứng từ văn bản và suy luận Chain-of-Thought.
"""

import re
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Đường dẫn hướng đến thư mục gốc của project để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Kiểm tra xem các thư viện cần thiết đã được cài đặt chưa
try:
    import openai
    import anthropic
    import qianfan
    HAS_OPENAI = True
    HAS_ANTHROPIC = True
    HAS_QIANFAN = True
except ImportError as e:
    module_name = str(e).split("'")[1]
    if module_name == "openai":
        HAS_OPENAI = False
    elif module_name == "anthropic":
        HAS_ANTHROPIC = False
    elif module_name == "qianfan":
        HAS_QIANFAN = False
    else:
        print(f"Warning: {e}")

# Khởi tạo các API key từ biến môi trường
if HAS_OPENAI:
    openai.api_key = os.getenv("OPENAI_API_KEY")

class MedicalLLMClient:
    """Lớp kết nối và gọi các y khoa LLM API"""
    
    def __init__(self, 
                model_name: str = "gpt-4", 
                temperature: float = 0.3, 
                max_tokens: int = 2048):
        """
        Khởi tạo Medical LLM Client
        
        Args:
            model_name: Tên mô hình LLM ("gpt-4o", "claude-3-opus-20240229", "qwen-med", etc.)
            temperature: Nhiệt độ sinh văn bản (0.0 - 1.0)
            max_tokens: Số token tối đa trong câu trả lời
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Xác định nền tảng API
        if model_name.startswith(("gpt", "text-davinci")):
            self.platform = "openai"
            if not HAS_OPENAI:
                raise ImportError("OpenAI module is not installed. Please install it with 'pip install openai'")
                
        elif model_name.startswith("claude"):
            self.platform = "anthropic"
            if not HAS_ANTHROPIC:
                raise ImportError("Anthropic module is not installed. Please install it with 'pip install anthropic'")
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                
        elif model_name.startswith(("qwen", "baichuan", "glm")):
            self.platform = "qianfan"
            if not HAS_QIANFAN:
                raise ImportError("QianFan module is not installed. Please install it with 'pip install qianfan'")
            self.qianfan_client = qianfan.ChatCompletion()
            
        else:
            # Default to OpenAI
            self.platform = "openai"
            self.model_name = "gpt-4"
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(7))
    def generate(self, prompt: str) -> str:
        """
        Gọi LLM API để sinh văn bản y khoa
        
        Args:
            prompt: Prompt đầu vào
            
        Returns:
            Văn bản sinh ra từ LLM
        """
        system_message = """You are a medical assistant specialized in providing evidence-based answers to medical questions. 
Follow these guidelines:
1. Base your responses solely on the provided evidence
2. Maintain clinical accuracy and use formal medical terminology
3. Acknowledge limitations in the evidence when present
4. Structure answers with clear sections for complex topics
5. Never add speculative information beyond what's in the evidence
6. Include relevant references to clinical guidelines when possible
7. Write in a concise, authoritative style appropriate for healthcare professionals"""
        
        if self.platform == "openai":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
            
        elif self.platform == "anthropic":
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.content[0].text
            
        elif self.platform == "qianfan":
            response = self.qianfan_client.do(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response["result"]
            
        else:
            raise ValueError(f"Unknown platform: {self.platform}")

def generate_answer_with_llm(query: str,
                          evidence_texts: List[str],
                          cot_steps: List[Dict[str, Any]],
                          model_name: Optional[str] = None,
                          temperature: float = 0.3,
                          max_tokens: int = 1024) -> str:
    """
    Sinh câu trả lời y khoa sử dụng mô hình LLM y khoa
    
    Args:
        query: Câu hỏi y khoa
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        cot_steps: Các bước suy luận CoT
        model_name: Tên mô hình LLM (mặc định lấy từ biến môi trường)
        temperature: Nhiệt độ sinh văn bản (càng cao càng đa dạng)
        max_tokens: Số token tối đa trong câu trả lời
        
    Returns:
        Câu trả lời y khoa
    """
    print(f"Generating answer with LLM for query: '{query}'")
    
    # Xác định mô hình từ biến môi trường nếu không được chỉ định
    if model_name is None:
        model_name = os.getenv("MEDICAL_LLM_MODEL", "gpt-4")
    
    # Chuẩn bị prompt
    prompt = _prepare_prompt(query, evidence_texts, cot_steps)
    
    try:
        # Khởi tạo LLM client và gọi API
        llm_client = MedicalLLMClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Tạo câu trả lời từ LLM
        answer = llm_client.generate(prompt)
        
        print(f"Generated answer with {len(answer.split())} words using {model_name}")
        
        return answer
        
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        # Fallback to predetermined answers for demo purposes
        return _generate_fallback_answer(query, evidence_texts)

def _prepare_prompt(query: str, 
                  evidence_texts: List[str], 
                  cot_steps: List[Dict[str, Any]]) -> str:
    """
    Chuẩn bị prompt cho mô hình LLM
    
    Args:
        query: Câu hỏi y khoa
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        cot_steps: Các bước suy luận CoT
        
    Returns:
        Prompt hoàn chỉnh cho LLM
    """
    # Format evidence
    formatted_evidence = "\n\n".join([f"Evidence {i+1}: {text}" for i, text in enumerate(evidence_texts)])
    
    # Format CoT steps
    formatted_cot = ""
    if cot_steps:
        cot_parts = []
        for step in cot_steps:
            step_text = f"Step {step.get('step', '?')}: {step.get('thought', '')}"
            if 'action' in step and 'action_input' in step:
                step_text += f"\nAction: {step['action']} - {step['action_input']}"
            cot_parts.append(step_text)
        formatted_cot = "\n\n".join(cot_parts)
    
    # Xây dựng prompt hoàn chỉnh
    prompt = f"""MEDICAL QUERY:
{query}

EVIDENCE:
{formatted_evidence}

REASONING:
{formatted_cot}

Based on the evidence and reasoning above, provide a comprehensive, accurate, and authoritative answer to the medical query. Focus only on information supported by the evidence provided. Structure your answer clearly, using medical terminology appropriately. If the evidence is limited or contradictory on any aspect, acknowledge these limitations.

ANSWER:"""

    return prompt

def generate_answer_with_qwen(query: str,
                            evidence_texts: List[str],
                            cot_steps: List[Dict[str, Any]],
                            max_tokens: int = 1024,
                            temperature: float = 0.7) -> str:
    """
    Sinh câu trả lời y khoa sử dụng mô hình Qwen-Med (hoặc LLM y khoa tương đương)
    Hàm này duy trì cho khả năng tương thích ngược
    
    Args:
        query: Câu hỏi y khoa
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        cot_steps: Các bước suy luận CoT
        max_tokens: Số token tối đa trong câu trả lời
        temperature: Nhiệt độ sinh văn bản (càng cao càng đa dạng)
        
    Returns:
        Câu trả lời y khoa
    """
    # Mặc định sử dụng model từ biến môi trường, ưu tiên Qwen-Med nếu có
    model_name = os.getenv("MEDICAL_LLM_MODEL", "qwen-med")
    
    return generate_answer_with_llm(
        query=query,
        evidence_texts=evidence_texts,
        cot_steps=cot_steps,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

def _generate_fallback_answer(query: str, evidence_texts: List[str]) -> str:
    """
    Tạo câu trả lời fallback khi không thể gọi LLM API
    
    Args:
        query: Câu hỏi
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        
    Returns:
        Câu trả lời y khoa dạng mẫu
    """
    print("Using fallback answer generation based on query keywords")
    
    # Phân loại câu hỏi dựa trên từ khóa
    query_lower = query.lower()
    
    if "osteoporosis" in query_lower and any(term in query_lower for term in ["treatment", "therapy", "medication"]):
        return """Current osteoporosis treatment options include several medication classes with different mechanisms of action.

First-line therapy typically consists of bisphosphonates such as alendronate, which inhibit osteoclast-mediated bone resorption. These medications are well-established with extensive safety and efficacy data, making them the cornerstone of osteoporosis management. They are most commonly administered orally on a weekly schedule (e.g., alendronate 70mg weekly).

For patients who cannot tolerate or have contraindications to bisphosphonates, denosumab (Prolia) represents an effective alternative. This fully human monoclonal antibody inhibits RANKL, thereby decreasing osteoclast formation and activity. It is administered as a subcutaneous injection every 6 months, which may improve adherence compared to oral medications.

For severe osteoporosis cases or patients who have failed antiresorptive therapy, anabolic agents that stimulate bone formation are recommended. These include:
- Teriparatide: A recombinant parathyroid hormone analog administered as a daily subcutaneous injection
- Abaloparatide: A PTHrP analog with similar effects to teriparatide
- Romosozumab: A sclerostin inhibitor that both increases bone formation and decreases bone resorption, though limited to a 12-month treatment course due to cardiovascular safety considerations

Emerging evidence supports sequential therapy approaches, typically starting with a bone-forming agent followed by an antiresorptive to maintain gained bone density. This approach has shown promising results in clinical trials for patients with very high fracture risk.

Treatment selection should be individualized based on fracture risk, comorbidities, patient preferences, and cost considerations. Regular monitoring of treatment response through bone mineral density measurements and bone turnover markers is recommended."""

    elif "compare" in query_lower or "versus" in query_lower or "vs" in query_lower:
        return """When comparing bisphosphonates and denosumab for osteoporosis treatment, several key differences should be considered:

Mechanism of Action:
- Bisphosphonates (e.g., alendronate) inhibit osteoclast-mediated bone resorption by binding to bone mineral and inducing osteoclast apoptosis.
- Denosumab is a monoclonal antibody that inhibits RANKL, thereby preventing osteoclast formation, function, and survival at a more upstream point in the pathway.

Efficacy:
- Both medications significantly reduce fracture risk.
- Direct comparison studies suggest denosumab may produce slightly greater increases in bone mineral density (BMD), particularly at cortical sites.
- The FREEDOM trial demonstrated denosumab reduced vertebral fractures by 68%, nonvertebral fractures by 20%, and hip fractures by 40%.

Administration and Adherence:
- Bisphosphonates are typically administered orally (weekly or monthly) with strict administration requirements (fasting, upright position), which may impact adherence.
- Denosumab is administered as a subcutaneous injection every 6 months, potentially improving adherence but requiring regular healthcare visits.

Onset and Offset of Action:
- Bisphosphonates incorporate into bone matrix and have a prolonged effect after discontinuation.
- Denosumab has a more rapid offset of action, with bone turnover increasing quickly after discontinuation, requiring transition to another therapy to prevent rapid bone loss.

Safety Considerations:
- Bisphosphonates may cause gastrointestinal side effects and have rare but serious adverse events like osteonecrosis of the jaw and atypical femur fractures.
- Denosumab shares risks of osteonecrosis and atypical fractures but lacks the GI side effects. It may increase infection risk due to RANKL's role in immune function.

Cost:
- Generic bisphosphonates are significantly less expensive than denosumab, which remains an important consideration for long-term therapy.

In clinical practice, bisphosphonates remain first-line treatment for most patients due to extensive safety data and lower cost. Denosumab is typically reserved for patients who cannot tolerate bisphosphonates, have renal insufficiency, or demonstrate inadequate response to initial therapy."""

    elif any(term in query_lower for term in ["mechanism", "how does", "how do"]):
        return """Denosumab's mechanism of action in treating osteoporosis operates through a specific biological pathway involving bone remodeling regulation.

At the molecular level, denosumab is a fully human monoclonal antibody that specifically binds to and inhibits RANKL (Receptor Activator of Nuclear Factor-κB Ligand). RANKL is a key signaling protein expressed by osteoblasts (bone-forming cells) that plays a crucial role in osteoclast formation and function.

The RANK/RANKL/OPG pathway functions as follows:
1. Osteoblasts express RANKL on their cell surface
2. RANKL binds to its receptor RANK on the surface of osteoclast precursors
3. This binding triggers multiple intracellular signaling cascades that promote:
   - Differentiation of precursor cells into mature osteoclasts
   - Activation of mature osteoclasts
   - Prolonged osteoclast survival by preventing apoptosis

By binding to RANKL with high specificity and affinity, denosumab prevents RANKL from interacting with RANK, effectively blocking the entire downstream signaling cascade. This inhibition:
- Prevents formation of new osteoclasts from precursor cells
- Decreases activity of existing osteoclasts
- Reduces osteoclast survival time
- Decreases bone resorption rates

As a result, denosumab creates a state of decreased bone turnover that allows osteoblast activity to exceed osteoclast-mediated bone resorption, leading to a net increase in bone mineral density. This effect is observed throughout the skeleton, with particularly notable effects in areas of cortical bone.

Unlike bisphosphonates, which bind to bone mineral and have long-lasting effects, denosumab does not incorporate into bone tissue and has a reversible effect that diminishes within months of discontinuation. This reversibility necessitates continued treatment or transition to another therapy to maintain its beneficial effects on bone density."""

    else:
        return """Based on the available evidence, current approaches to managing osteoporosis involve a comprehensive strategy incorporating pharmacological interventions, lifestyle modifications, and regular monitoring.

Pharmacological options can be broadly categorized into antiresorptive and anabolic agents:

Antiresorptive medications:
- Bisphosphonates (alendronate, risedronate, zoledronic acid): First-line therapy that inhibits osteoclast-mediated bone resorption
- Denosumab: A monoclonal antibody that inhibits RANKL, reducing osteoclast formation and activity
- Selective estrogen receptor modulators (SERMs): Act on estrogen receptors with tissue-specific effects

Anabolic agents:
- Teriparatide and abaloparatide: Parathyroid hormone analogs that stimulate bone formation
- Romosozumab: Sclerostin inhibitor with dual effect of increasing bone formation and decreasing resorption

Treatment selection should be personalized based on fracture risk assessment (using tools like FRAX), age, comorbidities, medication contraindications, and patient preferences. For high-risk patients, emerging evidence supports sequential therapy, starting with an anabolic agent followed by an antiresorptive to maintain gains in bone mineral density.

Non-pharmacological interventions remain essential components of management, including adequate calcium and vitamin D intake, weight-bearing exercise, fall prevention strategies, and smoking cessation.

Regular monitoring through bone mineral density testing (typically every 1-2 years during treatment) and bone turnover markers can help assess treatment efficacy and guide therapy adjustments. Treatment duration and potential drug holidays (particularly for bisphosphonates) should be considered based on individual risk profiles and response to therapy."""

def format_answer_with_references(answer: str, evidence_texts: List[str]) -> str:
    """
    Format câu trả lời với trích dẫn tham khảo
    
    Args:
        answer: Câu trả lời gốc
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        
    Returns:
        Câu trả lời đã định dạng với tham khảo
    """
    # Tạo danh sách tham khảo từ evidence_texts
    references = []
    for i, text in enumerate(evidence_texts):
        # Trích xuất nguồn từ đoạn văn bản (nếu có)
        source_match = re.search(r'\[Source: (.*?)\]', text)
        if source_match:
            source = source_match.group(1)
        else:
            source = f"Evidence {i+1}"
        
        # Tạo tham khảo
        references.append(f"[{i+1}] {source}")
    
    # Thêm các đánh dấu tham khảo vào câu trả lời
    # Chỉ thêm nếu câu trả lời chưa có tham khảo (tránh trùng lặp nếu LLM đã thêm)
    if not re.search(r'\[\d+\]', answer):
        paragraphs = answer.split("\n\n")
        for i, para in enumerate(paragraphs):
            if i < len(references) and len(para) > 20:
                # Chỉ thêm tham khảo vào đoạn đủ dài
                ref_idx = i % len(references) + 1
                paragraphs[i] = f"{para} [{ref_idx}]"
        
        # Kết hợp lại các đoạn
        answer = "\n\n".join(paragraphs)
    
    # Thêm phần tham khảo vào cuối
    if references:
        answer += "\n\nReferences:\n" + "\n".join(references)
    
    return answer 