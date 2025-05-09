"""
REFIND Verification Module
-----------------------
Module xác minh câu trả lời y khoa bằng phương pháp REFIND (Retrieval-Enhanced Factually
INcorrect Detection) để phát hiện và loại bỏ các thông tin không chính xác.
"""

import re
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import nltk

# Đảm bảo cài đặt dữ liệu tokenizer sentence cần thiết
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class REFINDVerifier:
    """Class thực hiện kiểm chứng thông tin theo phương pháp REFIND"""
    
    def __init__(self, 
                nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
                embedding_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
                device: Optional[str] = None):
        """
        Khởi tạo REFIND verifier
        
        Args:
            nli_model_name: Model NLI để kiểm tra entailment
            embedding_model_name: Model embedding cho semantic search
            device: Thiết bị để chạy model (None: tự động)
        """
        # Xác định thiết bị
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing REFIND Verifier on {self.device}")
        
        # Load NLI model
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.to(self.device)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Các nhãn NLI: (contradiction, entailment, neutral)
        self.nli_labels = ["contradiction", "entailment", "neutral"]
    
    def verify_answer(self, 
                    answer: str,
                    evidence_texts: List[str],
                    entailment_threshold: float = 0.5,
                    contradiction_threshold: float = 0.8) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Xác minh câu trả lời bằng cách kiểm tra tính nhất quán với bằng chứng
        
        Args:
            answer: Câu trả lời cần xác minh
            evidence_texts: Danh sách các đoạn văn bản bằng chứng
            entailment_threshold: Ngưỡng để một câu được coi là được hỗ trợ (entailed)
            contradiction_threshold: Ngưỡng để một câu được coi là mâu thuẫn
            
        Returns:
            Tuple(verified_answer, hallucinated_spans):
                - verified_answer: Câu trả lời đã được xác minh
                - hallucinated_spans: Danh sách các đoạn văn bản không có bằng chứng
        """
        # Chia câu trả lời thành các câu
        sentences = sent_tokenize(answer)
        print(f"Verifying {len(sentences)} sentences against {len(evidence_texts)} evidence pieces")
        
        # Tiền xử lý evidence
        flattened_evidence = " ".join(evidence_texts)
        evidence_sentences = sent_tokenize(flattened_evidence)
        
        # Tạo embedding cho evidence sentences
        evidence_embeddings = self.embedding_model.encode(
            evidence_sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        # Kiểm tra từng câu trong câu trả lời
        hallucinated_spans = []
        verified_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 10:  # Bỏ qua câu quá ngắn
                verified_sentences.append(sentence)
                continue
                
            # 1. Tìm các câu liên quan nhất trong bằng chứng
            claim_embedding = self.embedding_model.encode(
                sentence,
                show_progress_bar=False,
                convert_to_tensor=True
            )
            
            # Tính similarity và lấy top-k câu gần nhất
            similarities = torch.cosine_similarity(claim_embedding, evidence_embeddings)
            top_k = min(3, len(evidence_sentences))
            top_indices = torch.topk(similarities, k=top_k).indices.tolist()
            top_evidences = [evidence_sentences[idx] for idx in top_indices]
            
            # 2. Kiểm tra NLI giữa câu và evidence
            is_supported, nli_results = self._check_nli(
                claim=sentence,
                evidences=top_evidences,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold
            )
            
            # 3. Đánh dấu câu dựa trên kết quả
            if not is_supported:
                span = {
                    "text": sentence,
                    "index": i,
                    "nli_results": nli_results,
                    "reason": "Low evidence support or contradiction"
                }
                hallucinated_spans.append(span)
                
                # Đánh dấu câu không được xác minh
                verified_sentences.append(f"{sentence} [unverified]")
            else:
                verified_sentences.append(sentence)
        
        # Tạo câu trả lời đã xác minh
        verified_answer = " ".join(verified_sentences)
        
        return verified_answer, hallucinated_spans
    
    def _check_nli(self, 
                 claim: str, 
                 evidences: List[str],
                 entailment_threshold: float,
                 contradiction_threshold: float) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Kiểm tra NLI giữa claim và evidences
        
        Args:
            claim: Câu cần kiểm tra
            evidences: Danh sách các câu evidence
            entailment_threshold: Ngưỡng để một câu được coi là được hỗ trợ
            contradiction_threshold: Ngưỡng để một câu được coi là mâu thuẫn
            
        Returns:
            Tuple(is_supported, nli_results):
                - is_supported: True nếu claim được hỗ trợ bởi evidence
                - nli_results: Chi tiết kết quả NLI
        """
        nli_results = []
        max_entailment_score = 0.0
        max_contradiction_score = 0.0
        
        # Tokenize tất cả các cặp claim-evidence
        pairs = []
        for evidence in evidences:
            pairs.append((claim, evidence))
        
        # Batch processing NLI
        for premise, hypothesis in pairs:
            # Tokenize
            inputs = self.nli_tokenizer(
                premise, 
                hypothesis, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]
            
            # Lưu kết quả
            result = {
                "premise": premise,
                "hypothesis": hypothesis,
                "contradiction": float(scores[0]),
                "entailment": float(scores[1]),
                "neutral": float(scores[2])
            }
            nli_results.append(result)
            
            # Cập nhật điểm cao nhất
            max_entailment_score = max(max_entailment_score, result["entailment"])
            max_contradiction_score = max(max_contradiction_score, result["contradiction"])
        
        # Quyết định dựa trên ngưỡng
        is_contradicted = max_contradiction_score >= contradiction_threshold
        is_entailed = max_entailment_score >= entailment_threshold
        
        # Một câu được hỗ trợ khi:
        # 1. Có ít nhất một evidence hỗ trợ (entailment cao)
        # 2. Không có evidence nào mâu thuẫn mạnh
        is_supported = is_entailed and not is_contradicted
        
        return is_supported, nli_results
    
    def fix_hallucinated_content(self, 
                               answer: str, 
                               hallucinated_spans: List[Dict[str, Any]],
                               evidence_texts: List[str]) -> str:
        """
        Sửa chữa nội dung ảo giác trong câu trả lời
        
        Args:
            answer: Câu trả lời gốc
            hallucinated_spans: Danh sách các đoạn ảo giác
            evidence_texts: Danh sách các đoạn văn bản bằng chứng
            
        Returns:
            Câu trả lời đã được sửa chữa
        """
        if not hallucinated_spans:
            return answer
        
        fixed_answer = answer
        
        # Sắp xếp spans theo thứ tự ngược (từ cuối lên) để tránh ảnh hưởng index khi thay thế
        sorted_spans = sorted(hallucinated_spans, key=lambda x: x["index"], reverse=True)
        
        for span in sorted_spans:
            hallucinated_text = span["text"]
            
            # Xác định đoạn văn bản để thay thế
            replacement = self._find_replacement(hallucinated_text, evidence_texts)
            
            if replacement:
                # Thay thế bằng thông tin chính xác từ evidence
                fixed_answer = fixed_answer.replace(hallucinated_text, f"{replacement} [verified]")
            else:
                # Loại bỏ hoặc đánh dấu đoạn văn bản ảo giác
                fixed_answer = fixed_answer.replace(
                    hallucinated_text, 
                    "[Content removed due to insufficient evidence]"
                )
        
        return fixed_answer
    
    def _find_replacement(self, hallucinated_text: str, evidence_texts: List[str]) -> Optional[str]:
        """
        Tìm đoạn văn bản thay thế phù hợp từ evidence
        
        Args:
            hallucinated_text: Đoạn văn bản ảo giác
            evidence_texts: Danh sách các đoạn văn bản bằng chứng
            
        Returns:
            Đoạn văn bản thay thế hoặc None
        """
        # Chia evidence thành các câu
        evidence_sentences = []
        for evidence in evidence_texts:
            evidence_sentences.extend(sent_tokenize(evidence))
        
        # Mã hóa hallucinated_text và evidence_sentences
        text_embedding = self.embedding_model.encode(
            hallucinated_text,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        evidence_embeddings = self.embedding_model.encode(
            evidence_sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        
        # Tìm đoạn evidence liên quan nhất
        similarities = torch.cosine_similarity(text_embedding, evidence_embeddings)
        
        max_score = torch.max(similarities).item()
        if max_score > 0.7:  # Ngưỡng similarity để thay thế
            best_idx = torch.argmax(similarities).item()
            return evidence_sentences[best_idx]
        
        return None

def verify_answer_with_refind(draft_answer: str,
                            evidence_texts: List[str],
                            csr_threshold: float = 0.35) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Xác minh câu trả lời bằng phương pháp REFIND để phát hiện ảo giác (hallucination)
    
    Args:
        draft_answer: Câu trả lời gốc từ LLM
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        csr_threshold: Ngưỡng tỷ lệ nhạy cảm ngữ cảnh (CSR)
        
    Returns:
        Tuple(verified_answer, hallucinated_spans):
            - verified_answer: Câu trả lời đã được xác minh
            - hallucinated_spans: Danh sách các đoạn văn bản không có bằng chứng
    """
    try:
        # Tạo REFIND verifier
        verifier = REFINDVerifier()
        
        # Xác minh câu trả lời
        verified_answer, hallucinated_spans = verifier.verify_answer(
            answer=draft_answer,
            evidence_texts=evidence_texts,
            entailment_threshold=0.5,  # Tuỳ chỉnh ngưỡng entailment
            contradiction_threshold=0.8  # Tuỳ chỉnh ngưỡng contradiction
        )
        
        # Fix các đoạn ảo giác
        if hallucinated_spans:
            verified_answer = verifier.fix_hallucinated_content(
                verified_answer,
                hallucinated_spans,
                evidence_texts
            )
        
        print(f"Verified answer - found {len(hallucinated_spans)} potential hallucinations")
        
        return verified_answer, hallucinated_spans
        
    except Exception as e:
        print(f"Error in REFIND verification: {e}")
        # Fallback to context-sensitive ratio analysis
        return _fallback_csr_verification(draft_answer, evidence_texts, csr_threshold)

def _fallback_csr_verification(draft_answer: str,
                             evidence_texts: List[str],
                             csr_threshold: float = 0.35) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Phương pháp fallback dựa trên Context Sensitivity Ratio khi NLI không khả dụng
    
    Args:
        draft_answer: Câu trả lời gốc từ LLM
        evidence_texts: Danh sách các đoạn văn bản bằng chứng
        csr_threshold: Ngưỡng tỷ lệ nhạy cảm ngữ cảnh (CSR)
        
    Returns:
        Tuple(verified_answer, hallucinated_spans)
    """
    print(f"Fallback to CSR verification, threshold: {csr_threshold}")
    
    # Chia câu trả lời thành các câu riêng biệt
    sentences = sent_tokenize(draft_answer)
    
    # Phát hiện các câu có khả năng ảo giác
    hallucinated_spans = []
    verified_sentences = []
    
    # Tạo corpus từ evidence
    corpus = " ".join(evidence_texts).lower()
    
    for i, sentence in enumerate(sentences):
        # Tính CSR (Context Sensitivity Ratio)
        csr_score = _calculate_csr(sentence, corpus)
        
        if csr_score < csr_threshold:
            # Câu có khả năng chứa ảo giác
            span = {
                "text": sentence,
                "index": i,
                "csr_score": csr_score,
                "reason": f"Low CSR score: {csr_score:.2f} < threshold: {csr_threshold:.2f}"
            }
            hallucinated_spans.append(span)
            
            # Đánh dấu câu có vấn đề
            verified_sentences.append(f"{sentence} [unverified]")
        else:
            verified_sentences.append(sentence)
    
    # Tạo câu trả lời đã xác minh
    verified_answer = " ".join(verified_sentences)
    
    # In các thông tin xác minh
    print(f"Verified {len(sentences)} sentences with CSR")
    print(f"Found {len(hallucinated_spans)} potential hallucinations")
    
    return verified_answer, hallucinated_spans

def _calculate_csr(sentence: str, corpus: str) -> float:
    """
    Tính toán Context Sensitivity Ratio (CSR) cho một câu
    
    CSR = (số từ/cụm từ có trong corpus) / (tổng số từ trong câu)
    
    Args:
        sentence: Câu cần tính CSR
        corpus: Văn bản corpus tổng hợp từ evidence
        
    Returns:
        Điểm CSR (0.0 - 1.0)
    """
    # Tiền xử lý câu
    sentence = sentence.lower()
    
    # Tách các từ (bỏ qua dấu câu)
    words = re.findall(r'\b\w+\b', sentence)
    
    if not words:
        return 1.0  # Câu rỗng hoặc chỉ có dấu câu
    
    # Đếm số từ có trong corpus
    words_in_corpus = 0
    for word in words:
        if word in corpus:
            words_in_corpus += 1
    
    # Tính tỷ lệ CSR
    csr = words_in_corpus / len(words)
    
    # Kiểm tra các cụm từ quan trọng
    important_phrases = _extract_important_phrases(sentence)
    phrases_found = 0
    
    for phrase in important_phrases:
        if phrase.lower() in corpus:
            phrases_found += 1
    
    # Điều chỉnh CSR dựa trên cụm từ quan trọng
    if important_phrases:
        phrase_csr = phrases_found / len(important_phrases)
        # Trọng số cho cụm từ quan trọng cao hơn
        csr = 0.7 * csr + 0.3 * phrase_csr
    
    return csr

def _extract_important_phrases(sentence: str) -> List[str]:
    """
    Trích xuất các cụm từ quan trọng từ câu
    
    Args:
        sentence: Câu cần trích xuất
        
    Returns:
        Danh sách các cụm từ quan trọng
    """
    # DEMO: Trích xuất các cụm từ y khoa quan trọng
    # Thực tế sẽ dùng NER hoặc phương pháp phức tạp hơn
    
    important_phrases = []
    
    # Mẫu cụm từ y khoa quan trọng
    medical_patterns = [
        r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|mmol|IU)\b',  # Liều lượng
        r'\b[A-Z][a-z]+(?:mab|zumab|limab|umab)\b',      # Thuốc kháng thể
        r'\bbisphosphonate[s]?\b',                       # Nhóm thuốc
        r'\b(?:anti|de)[a-z]+(?:ase|ine)\b',             # Enzyme
        r'\b[A-Z]+[A-Z0-9]*\b',                          # Viết tắt y khoa
        r'\b[A-Z][a-z]+(?:one|ol|ine|ide|ate)\b'         # Tên thuốc phổ biến
    ]
    
    # Tìm các cụm từ theo mẫu
    for pattern in medical_patterns:
        matches = re.finditer(pattern, sentence)
        for match in matches:
            important_phrases.append(match.group())
    
    # Cụm danh từ y khoa cụ thể
    medical_terms = [
        "osteoporosis", "bone density", "fracture risk", "bone mineral density",
        "bone resorption", "bone formation", "osteoclast", "osteoblast",
        "RANKL", "denosumab", "alendronate", "teriparatide", "romosozumab",
        "bisphosphonate", "monoclonal antibody", "parathyroid hormone",
        "sclerostin inhibitor", "antiresorptive", "anabolic", "estrogen receptor"
    ]
    
    # Tìm các cụm từ y khoa cụ thể
    for term in medical_terms:
        if term.lower() in sentence.lower():
            # Tìm vị trí chính xác và giữ nguyên cách viết hoa
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            matches = pattern.finditer(sentence)
            for match in matches:
                important_phrases.append(match.group())
    
    return important_phrases 