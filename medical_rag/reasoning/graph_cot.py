"""
Graph-CoT Reasoning Module
-----------------------
Module thực hiện suy luận dạng Chain-of-Thought (CoT) trên đồ thị tri thức y khoa
để hướng dẫn quá trình trả lời câu hỏi y tế.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
import os
from neo4j import GraphDatabase
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Cấu hình API key từ biến môi trường
openai.api_key = os.getenv("OPENAI_API_KEY")

class Neo4jGraphConnector:
    """Class kết nối và truy vấn đồ thị tri thức Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chạy Cypher query trên Neo4j và trả về kết quả"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def get_node_by_id(self, node_id: int) -> Dict[str, Any]:
        """Lấy thông tin về node theo ID"""
        query = """
        MATCH (n) WHERE id(n) = $node_id
        RETURN n
        """
        results = self.run_query(query, {"node_id": node_id})
        if results:
            return results[0]["n"]
        return {}
    
    def get_node_neighbors(self, node_id: int, rel_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lấy các node kề và quan hệ"""
        if rel_type:
            query = f"""
            MATCH (n)-[r:{rel_type}]->(m) WHERE id(n) = $node_id
            RETURN type(r) as relation, id(m) as neighbor_id, labels(m) as types, m.name as name
            """
        else:
            query = """
            MATCH (n)-[r]->(m) WHERE id(n) = $node_id
            RETURN type(r) as relation, id(m) as neighbor_id, labels(m) as types, m.name as name
            """
        return self.run_query(query, {"node_id": node_id})
    
    def find_paths(self, start_id: int, end_id: int, max_depth: int = 3) -> List[List[Dict[str, Any]]]:
        """Tìm đường đi giữa hai node"""
        query = """
        MATCH p = shortestPath((start)-[*..%d]->(end))
        WHERE id(start) = $start_id AND id(end) = $end_id
        RETURN [n IN nodes(p) | {id: id(n), labels: labels(n), name: n.name}] AS path,
               [r IN relationships(p) | {type: type(r)}] AS rels
        """ % max_depth
        
        results = self.run_query(query, {"start_id": start_id, "end_id": end_id})
        
        paths = []
        for result in results:
            nodes = result["path"]
            rels = result["rels"]
            path = []
            
            # Kết hợp nodes và relationships để tạo đường đi
            for i in range(len(nodes) - 1):
                path.append({
                    "from": nodes[i],
                    "relation": rels[i]["type"],
                    "to": nodes[i+1]
                })
            
            paths.append(path)
        
        return paths
    
    def execute_graph_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Thực thi Cypher query và trả về kết quả"""
        try:
            return self.run_query(cypher_query)
        except Exception as e:
            print(f"Error executing Cypher query: {e}")
            return []

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(7))
def call_llm_api(prompt: str, 
                model: str = "gpt-4", 
                temperature: float = 0.7, 
                max_tokens: int = 1000) -> str:
    """
    Gọi LLM API để sinh văn bản
    
    Args:
        prompt: Prompt đầu vào
        model: Model sử dụng ("gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", etc.)
        temperature: Nhiệt độ sinh văn bản (0.0 - 1.0)
        max_tokens: Số token tối đa sinh ra
        
    Returns:
        Văn bản sinh ra từ LLM
    """
    # Xác định nền tảng API dựa vào tên model
    if model.startswith("gpt"):
        # OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a medical reasoning assistant. Think through the given query step by step using the evidence from the medical knowledge graph."},
                     {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    elif model.startswith("claude"):
        # Anthropic API
        try:
            from anthropic import Anthropic
            anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            response = anthropic.messages.create(
                model=model,
                system="You are a medical reasoning assistant. Think through the given query step by step using the evidence from the medical knowledge graph.",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except ImportError:
            raise ImportError("The anthropic library is not installed. Please install it with 'pip install anthropic'.")
    
    elif model.startswith("qwen") or model.startswith("baichuan") or model.startswith("glm"):
        # Zhipu AI / QianFan API
        try:
            import qianfan
            
            client = qianfan.ChatCompletion()
            response = client.do(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a medical reasoning assistant. Think through the given query step by step using the evidence from the medical knowledge graph."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response["result"]
        except ImportError:
            raise ImportError("The qianfan library is not installed. Please install it with 'pip install qianfan'.")
    
    else:
        # Default to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a medical reasoning assistant. Think through the given query step by step using the evidence from the medical knowledge graph."},
                     {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

class GraphCoTReasoner:
    """Class thực hiện suy luận Chain-of-Thought trên đồ thị tri thức y khoa"""
    
    def __init__(self, 
                neo4j_uri: str = None, 
                neo4j_user: str = None, 
                neo4j_password: str = None,
                llm_model: str = "gpt-4"):
        """
        Khởi tạo reasoner
        
        Args:
            neo4j_uri: URI của Neo4j database
            neo4j_user: Tên đăng nhập Neo4j
            neo4j_password: Mật khẩu Neo4j
            llm_model: Model language model sử dụng cho suy luận
        """
        # Kết nối đồ thị Neo4j
        if neo4j_uri and neo4j_user and neo4j_password:
            self.graph = Neo4jGraphConnector(neo4j_uri, neo4j_user, neo4j_password)
        else:
            # Lấy từ biến môi trường nếu không cung cấp
            self.graph = Neo4jGraphConnector(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
        
        self.llm_model = llm_model
    
    def reason(self, 
              query: str, 
              subgraph_nodes: List[int],
              subgraph_edges: List[Tuple[int, str, int]],
              max_steps: int = 5,
              temperature: float = 0.7) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Thực hiện suy luận CoT dựa trên đồ thị
        
        Args:
            query: Câu hỏi y khoa
            subgraph_nodes: Danh sách các node ID trong tiểu đồ thị
            subgraph_edges: Danh sách các cạnh trong tiểu đồ thị (src, rel, dst)
            max_steps: Số bước suy luận tối đa
            temperature: Nhiệt độ cho LLM
            
        Returns:
            Tuple(cot_steps, evidence_nodes):
                - cot_steps: Danh sách các bước suy luận
                - evidence_nodes: Danh sách các node làm bằng chứng
        """
        print(f"Starting Graph-CoT reasoning for query: '{query}'")
        
        # 1. Xây dựng đồ thị context từ subgraph
        graph_context = self._build_graph_context(subgraph_nodes, subgraph_edges)
        
        # 2. Xác định loại câu hỏi và chiến lược suy luận
        question_type, reasoning_strategy = self._analyze_question_type(query)
        
        # 3. Tạo prompt cho CoT
        cot_prompt = self._create_cot_prompt(query, graph_context, question_type, reasoning_strategy)
        
        # 4. Gọi LLM để thực hiện suy luận
        cot_output = call_llm_api(
            prompt=cot_prompt,
            model=self.llm_model,
            temperature=temperature,
            max_tokens=1500
        )
        
        # 5. Parse output để lấy các bước suy luận
        cot_steps = self._parse_cot_output(cot_output, max_steps)
        
        # 6. Trích xuất các node bằng chứng từ các bước suy luận
        evidence_nodes = self._extract_evidence_nodes(cot_steps, subgraph_nodes)
        
        print(f"Generated {len(cot_steps)} reasoning steps with {len(evidence_nodes)} evidence nodes")
        
        return cot_steps, evidence_nodes
    
    def _build_graph_context(self, 
                           subgraph_nodes: List[int],
                           subgraph_edges: List[Tuple[int, str, int]]) -> str:
        """Xây dựng context string từ tiểu đồ thị cho LLM"""
        
        # Lấy thông tin về các node
        node_info = {}
        for node_id in subgraph_nodes:
            try:
                node_data = self.graph.get_node_by_id(node_id)
                if node_data:
                    node_info[node_id] = node_data
            except Exception as e:
                # Nếu không kết nối được Neo4j, tạo thông tin node giả
                node_info[node_id] = self._create_mock_node(node_id)
        
        # Tạo mô tả đồ thị
        nodes_context = [f"Node {node_id}: {self._format_node(node_data)}" 
                         for node_id, node_data in node_info.items()]
        
        edges_context = []
        for src, rel, dst in subgraph_edges:
            if src in node_info and dst in node_info:
                src_name = node_info[src].get("name", f"Node_{src}")
                dst_name = node_info[dst].get("name", f"Node_{dst}")
                edges_context.append(f"{src_name} --[{rel}]--> {dst_name}")
        
        # Kết hợp mô tả
        graph_context = "KNOWLEDGE GRAPH INFORMATION:\n\n"
        graph_context += "Nodes:\n" + "\n".join(nodes_context) + "\n\n"
        graph_context += "Relationships:\n" + "\n".join(edges_context)
        
        return graph_context
    
    def _create_mock_node(self, node_id: int) -> Dict[str, Any]:
        """Tạo thông tin node giả khi không kết nối được với Neo4j"""
        # Phân loại node dựa trên ID (Như trong seed_select.py demo)
        node_type = "Concept"
        node_name = f"Node_{node_id}"
        
        if 1001 <= node_id <= 1099:
            node_type = "Drug"
            if node_id == 1001:
                node_name = "Alendronate"
            elif node_id == 1002:
                node_name = "Denosumab"
            elif node_id == 1003:
                node_name = "Teriparatide"
            elif node_id == 1004:
                node_name = "Romosozumab"
            elif node_id == 1005:
                node_name = "Zoledronic acid"
                
        elif 1100 <= node_id <= 1199:
            node_type = "Disease"
            if node_id == 1101:
                node_name = "Osteoporosis"
            elif node_id == 1102:
                node_name = "Osteopenia"
            elif node_id == 1103:
                node_name = "Fracture"
                
        elif 1200 <= node_id <= 1299:
            node_type = "Mechanism"
            if node_id == 1201:
                node_name = "Bone resorption"
            elif node_id == 1202:
                node_name = "Bone formation"
            elif node_id == 1203:
                node_name = "RANKL inhibition"
                
        elif 1300 <= node_id <= 1399:
            node_type = "Anatomy"
            if node_id == 1301:
                node_name = "Bone"
            elif node_id == 1302:
                node_name = "Skeleton"
        
        return {
            "id": node_id,
            "labels": [node_type],
            "name": node_name,
            "type": node_type,
            "desc": f"Mock {node_type.lower()} node with ID {node_id}"
        }
    
    def _format_node(self, node_data: Dict[str, Any]) -> str:
        """Format thông tin node thành string dễ đọc"""
        node_type = node_data.get("type", "")
        if not node_type and "labels" in node_data:
            node_type = node_data["labels"][0] if node_data["labels"] else ""
            
        node_name = node_data.get("name", "")
        node_desc = node_data.get("desc", "")
        
        return f"[{node_type}] {node_name}" + (f" - {node_desc}" if node_desc else "")
    
    def _analyze_question_type(self, query: str) -> Tuple[str, str]:
        """Phân tích loại câu hỏi và chiến lược suy luận"""
        query_lower = query.lower()
        
        # Phân loại câu hỏi
        if "treatment" in query_lower or "therapy" in query_lower or "medication" in query_lower:
            question_type = "treatment"
            reasoning_strategy = "treatment_analysis"
            
        elif "compare" in query_lower or "versus" in query_lower or "vs" in query_lower:
            question_type = "comparison"
            reasoning_strategy = "comparative_analysis"
            
        elif "mechanism" in query_lower or "how does" in query_lower or "how do" in query_lower:
            question_type = "mechanism"
            reasoning_strategy = "mechanism_explanation"
            
        elif "cause" in query_lower or "risk factor" in query_lower or "lead to" in query_lower:
            question_type = "causation"
            reasoning_strategy = "causal_analysis"
            
        elif "diagnose" in query_lower or "test" in query_lower or "detect" in query_lower:
            question_type = "diagnostic"
            reasoning_strategy = "diagnostic_approach"
            
        elif "prevent" in query_lower or "reduce risk" in query_lower:
            question_type = "prevention"
            reasoning_strategy = "preventive_approach"
            
        else:
            question_type = "general"
            reasoning_strategy = "general_analysis"
        
        return question_type, reasoning_strategy
    
    def _create_cot_prompt(self, 
                          query: str, 
                          graph_context: str,
                          question_type: str,
                          reasoning_strategy: str) -> str:
        """Tạo prompt cho LLM để thực hiện suy luận CoT"""
        
        # Phần instruction cố định
        instructions = """You are a medical reasoning system analyzing a medical question using knowledge graph information.

Your task is to perform chain-of-thought reasoning on the provided knowledge graph to answer the medical query.

For each step:
1. Analyze the relevant nodes and relationships
2. Draw logical conclusions from the connections
3. Explain your reasoning in detail
4. Support your reasoning with evidence from the knowledge graph

Format your answer as a series of numbered steps (Step 1, Step 2, etc.), with each step containing:
- Your thought process
- The action you're taking (ANALYZE, QUERY, CONNECT, INFER, CONCLUDE)
- The specific nodes or relationships you're using as evidence

End with a clear conclusion that answers the original question based on your reasoning.
"""

        # Tùy chỉnh hướng dẫn dựa trên loại câu hỏi
        question_guidance = ""
        if question_type == "treatment":
            question_guidance = """This question is about medical treatments. Focus on:
- Different treatment options available
- Mechanisms of action for each treatment
- Effectiveness and appropriate use cases
- First-line vs. alternative treatments
- Potential side effects (if mentioned in the knowledge graph)
"""
        elif question_type == "comparison":
            question_guidance = """This question requires comparing different treatments or approaches. Focus on:
- The key differences in mechanisms
- Relative efficacy and safety profiles
- Specific indications for each option
- Benefits and limitations of each approach
"""
        elif question_type == "mechanism":
            question_guidance = """This question is about mechanisms of action. Focus on:
- The biological pathway involved
- How the drug/treatment affects the target
- Downstream effects of the mechanism
- Connection between mechanism and clinical outcomes
"""
        
        # Kết hợp tất cả để tạo prompt hoàn chỉnh
        prompt = f"""{instructions}

{question_guidance}

MEDICAL QUERY: {query}

{graph_context}

Now, provide your step-by-step chain-of-thought reasoning to answer this query:
"""
        
        return prompt
    
    def _parse_cot_output(self, cot_output: str, max_steps: int) -> List[Dict[str, Any]]:
        """Parse output từ LLM để lấy các bước suy luận CoT"""
        # Tách các bước từ output
        import re
        
        # Pattern để tìm các bước
        step_pattern = r"Step\s+(\d+):?\s*(.*?)(?=Step\s+\d+:|$)"
        matches = re.finditer(step_pattern, cot_output, re.DOTALL)
        
        cot_steps = []
        for match in matches:
            step_number = int(match.group(1))
            step_content = match.group(2).strip()
            
            # Tìm action trong bước
            action_match = re.search(r"Action:?\s*([A-Z]+)", step_content)
            action = action_match.group(1) if action_match else "ANALYZE"
            
            # Tìm input cho action
            action_input_match = re.search(r"Action:?\s*[A-Z]+\s*-\s*(.*?)(?:\n|$)", step_content)
            action_input = action_input_match.group(1).strip() if action_input_match else ""
            
            # Phần còn lại là thought
            thought = re.sub(r"Action:?\s*[A-Z]+(?:\s*-\s*.*?)?(?:\n|$)", "", step_content).strip()
            
            cot_steps.append({
                "step": step_number,
                "thought": thought,
                "action": action,
                "action_input": action_input
            })
            
            if len(cot_steps) >= max_steps:
                break
        
        # Nếu không tìm thấy các bước rõ ràng, chia cả đoạn thành các bước
        if not cot_steps:
            paragraphs = cot_output.split("\n\n")
            for i, para in enumerate(paragraphs[:max_steps]):
                cot_steps.append({
                    "step": i + 1,
                    "thought": para.strip(),
                    "action": "ANALYZE",
                    "action_input": ""
                })
        
        return cot_steps
    
    def _extract_evidence_nodes(self, 
                              cot_steps: List[Dict[str, Any]], 
                              subgraph_nodes: List[int]) -> List[int]:
        """Trích xuất các node bằng chứng từ các bước suy luận"""
        evidence_nodes = []
        
        # Tạo lookup các tên node từ subgraph_nodes
        node_names = {}
        for node_id in subgraph_nodes:
            try:
                node_data = self.graph.get_node_by_id(node_id)
                if node_data and "name" in node_data:
                    node_name = node_data["name"].lower()
                    node_names[node_name] = node_id
            except Exception:
                # Nếu không kết nối được Neo4j, tạo thông tin từ mock node
                mock_node = self._create_mock_node(node_id)
                node_name = mock_node["name"].lower()
                node_names[node_name] = node_id
        
        # Duyệt qua các bước suy luận và tìm tham chiếu đến các node
        for step in cot_steps:
            # Kết hợp thought và action_input để tìm kiếm
            text = (step["thought"] + " " + step["action_input"]).lower()
            
            # Tìm các tham chiếu đến node theo tên
            for node_name, node_id in node_names.items():
                if node_name.lower() in text:
                    evidence_nodes.append(node_id)
        
        # Đảm bảo uniqueness và thứ tự
        evidence_nodes = list(dict.fromkeys(evidence_nodes))
        
        return evidence_nodes

def reasoning_with_graph_cot(query: str,
                          subgraph_nodes: List[int],
                          subgraph_edges: List[Tuple[int, str, int]],
                          max_steps: int = 5) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Thực hiện suy luận chuỗi suy nghĩ trên đồ thị tri thức y khoa
    
    Args:
        query: Câu hỏi y khoa
        subgraph_nodes: Danh sách các nút trong tiểu đồ thị
        subgraph_edges: Danh sách các cạnh trong tiểu đồ thị
        max_steps: Số bước suy luận tối đa
        
    Returns:
        Tuple(cot_steps, evidence_nodes):
            - cot_steps: Danh sách các bước suy luận
            - evidence_nodes: Danh sách các nút làm bằng chứng
    """
    # Lấy thông tin kết nối Neo4j từ biến môi trường
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Lấy model LLM từ biến môi trường hoặc dùng default
    llm_model = os.getenv("COT_LLM_MODEL", "gpt-4")
    
    # Khởi tạo reasoner
    reasoner = GraphCoTReasoner(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        llm_model=llm_model
    )
    
    # Thực hiện suy luận
    try:
        cot_steps, evidence_nodes = reasoner.reason(
            query=query,
            subgraph_nodes=subgraph_nodes,
            subgraph_edges=subgraph_edges,
            max_steps=max_steps
        )
    except Exception as e:
        print(f"Error during graph reasoning: {e}")
        # Fallback to demo if errors occur
        cot_steps, evidence_nodes = _demo_reasoning(query, subgraph_nodes, max_steps)
    
    return cot_steps, evidence_nodes

def _demo_reasoning(query: str, 
                  subgraph_nodes: List[int], 
                  max_steps: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Phương thức backup tạo các bước suy luận mẫu khi kết nối thất bại
    """
    print("[DEMO] Using demo reasoning due to connection issues")
    
    # Tạo các bước suy luận mẫu
    if "osteoporosis" in query.lower() and any(term in query.lower() for term in ["treatment", "therapy", "medication"]):
        # Trường hợp: Query về điều trị osteoporosis
        cot_steps = [
            {
                "step": 1, 
                "thought": "Câu hỏi là về điều trị osteoporosis. Tôi cần xác định các phương pháp điều trị hiện có.",
                "action": "QUERY",
                "action_input": "Tìm tất cả các thuốc điều trị osteoporosis"
            },
            {
                "step": 2,
                "thought": "Từ đồ thị, tôi thấy có nhiều loại thuốc điều trị osteoporosis, chia thành các nhóm chính: bisphosphonates (alendronate, zoledronic acid), kháng thể đơn dòng (denosumab), và thuốc tạo xương (teriparatide, romosozumab).",
                "action": "ANALYZE",
                "action_input": "Phân loại các thuốc điều trị osteoporosis"
            },
            {
                "step": 3,
                "thought": "Mỗi loại thuốc có cơ chế hoạt động khác nhau. Bisphosphonates và denosumab ức chế tiêu xương, trong khi teriparatide và romosozumab kích thích tạo xương.",
                "action": "CONNECT",
                "action_input": "Kết nối thuốc với cơ chế hoạt động"
            },
            {
                "step": 4,
                "thought": "Theo hướng dẫn điều trị, bisphosphonates như alendronate thường được dùng làm liệu pháp đầu tay do có nhiều dữ liệu về hiệu quả và an toàn. Denosumab được dùng cho bệnh nhân không đáp ứng hoặc không dung nạp bisphosphonates.",
                "action": "INFER",
                "action_input": "Suy luận về thứ tự ưu tiên điều trị"
            },
            {
                "step": 5,
                "thought": "Với bệnh nhân osteoporosis nặng hoặc có nguy cơ gãy xương cao, thuốc tạo xương như teriparatide hoặc romosozumab có thể được ưu tiên, đặc biệt là khi các thuốc khác không hiệu quả.",
                "action": "CONCLUDE",
                "action_input": "Kết luận về chiến lược điều trị tối ưu"
            }
        ]
        
        # Chọn các node liên quan
        evidence_nodes = [node for node in subgraph_nodes if 1001 <= node <= 1005 or node == 1101]
        
    elif "compare" in query.lower() or "versus" in query.lower() or "vs" in query.lower():
        # Trường hợp: Query so sánh
        cot_steps = [
            {
                "step": 1,
                "thought": "Câu hỏi yêu cầu so sánh các phương pháp điều trị. Tôi cần xác định các thuốc được đề cập.",
                "action": "EXTRACT",
                "action_input": "Trích xuất tên các thuốc để so sánh"
            },
            {
                "step": 2,
                "thought": "Từ đồ thị, tôi thấy cần so sánh bisphosphonates (như alendronate) với denosumab. Cả hai đều điều trị osteoporosis nhưng có cơ chế khác nhau.",
                "action": "ANALYZE",
                "action_input": "Phân tích cơ chế hoạt động của các thuốc"
            },
            {
                "step": 3,
                "thought": "Alendronate ức chế tiêu xương bằng cách gắn vào khoáng chất xương và gây apoptosis tế bào hủy xương. Denosumab ức chế RANKL, ngăn sự hình thành, hoạt động và tồn tại của tế bào hủy xương ở điểm ngược dòng hơn trong con đường.",
                "action": "COMPARE",
                "action_input": "So sánh cơ chế phân tử"
            },
            {
                "step": 4,
                "thought": "Về hiệu quả, cả hai thuốc đều giảm đáng kể nguy cơ gãy xương. Nghiên cứu so sánh trực tiếp cho thấy denosumab có thể tạo ra sự tăng mật độ xương lớn hơn một chút, đặc biệt là ở các vị trí xương vỏ.",
                "action": "INFER",
                "action_input": "Suy luận về hiệu quả lâm sàng"
            },
            {
                "step": 5,
                "thought": "Về độ an toàn và tiện lợi: alendronate dùng uống hàng tuần, yêu cầu ở tư thế thẳng đứng. Denosumab tiêm dưới da 6 tháng/lần, thuận tiện hơn nhưng đắt hơn. Bisphosphonates có thể gây tác dụng phụ tiêu hóa, trong khi denosumab có thể tăng nhẹ nguy cơ nhiễm trùng.",
                "action": "CONCLUDE",
                "action_input": "Kết luận về so sánh tổng thể"
            }
        ]
        
        # Chọn các node liên quan
        evidence_nodes = [1001, 1002, 1101, 1201, 1203]
        
    else:
        # Trường hợp mặc định
        cot_steps = [
            {
                "step": 1,
                "thought": "Câu hỏi liên quan đến tìm hiểu về thuốc mới điều trị osteoporosis. Tôi cần tìm trong đồ thị các thuốc được phân loại là 'mới' hoặc 'tiên tiến'.",
                "action": "QUERY",
                "action_input": "Tìm thuốc mới điều trị osteoporosis"
            },
            {
                "step": 2,
                "thought": "Từ đồ thị tri thức, tôi tìm thấy một số thuốc tương đối mới bao gồm denosumab (Prolia), teriparatide, abaloparatide và romosozumab (Evenity).",
                "action": "ANALYZE",
                "action_input": "Phân tích các thuốc mới"
            },
            {
                "step": 3,
                "thought": "Denosumab là kháng thể đơn dòng ức chế RANKL, do đó giảm hình thành và hoạt động của tế bào hủy xương. Thuốc được tiêm dưới da 6 tháng/lần.",
                "action": "DESCRIBE",
                "action_input": "Mô tả denosumab"
            },
            {
                "step": 4,
                "thought": "Romosozumab là thuốc ức chế sclerostin, vừa tăng tạo xương vừa giảm tiêu xương, nhưng giới hạn thời gian điều trị 12 tháng do những lo ngại về an toàn tim mạch.",
                "action": "DESCRIBE",
                "action_input": "Mô tả romosozumab"
            },
            {
                "step": 5,
                "thought": "Các nghiên cứu đang đánh giá liệu pháp tuần tự, thường bắt đầu bằng thuốc tạo xương sau đó đến thuốc chống tiêu xương để duy trì mật độ xương đã tăng. Cách tiếp cận này đã cho kết quả đầy hứa hẹn trong các thử nghiệm lâm sàng cho bệnh nhân có nguy cơ gãy xương rất cao.",
                "action": "CONCLUDE",
                "action_input": "Kết luận về liệu pháp mới"
            }
        ]
        
        # Chọn các node liên quan
        evidence_nodes = [1002, 1003, 1004, 1101, 1201, 1202, 1203]
    
    # Giới hạn số bước suy luận
    cot_steps = cot_steps[:max_steps]
    
    return cot_steps, evidence_nodes 