"""
MixPR Router Module
----------------
Module định tuyến đồ thị y khoa bằng thuật toán MixPR (Mixed Personalized PageRank)
để xác định tiểu đồ thị liên quan đến truy vấn.
"""

import numpy as np
import os
import sys
import warnings
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from collections import defaultdict
import networkx as nx

# Đường dẫn hướng đến thư mục gốc của project để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Kiểm tra Neo4j
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    warnings.warn("Neo4j Driver not found. Neo4j-based MixPR will not be available.")

class MixPRRouter:
    """Class triển khai thuật toán Mixed Personalized PageRank"""
    
    def __init__(self, 
                use_neo4j: bool = False,
                neo4j_uri: Optional[str] = None,
                neo4j_user: Optional[str] = None,
                neo4j_password: Optional[str] = None,
                in_memory_graph: Optional[nx.DiGraph] = None):
        """
        Khởi tạo MixPR Router
        
        Args:
            use_neo4j: Có sử dụng Neo4j hay không
            neo4j_uri: URI kết nối Neo4j (nếu dùng Neo4j)
            neo4j_user: Tên đăng nhập Neo4j
            neo4j_password: Mật khẩu Neo4j
            in_memory_graph: Đồ thị NetworkX trong bộ nhớ (nếu không dùng Neo4j)
        """
        self.use_neo4j = use_neo4j
        self.neo4j_driver = None
        self.graph = in_memory_graph
        
        # Kết nối Neo4j nếu cần
        if use_neo4j:
            if not HAS_NEO4J:
                raise ImportError("Neo4j Driver not installed. Install with 'pip install neo4j'")
                
            # Lấy thông tin kết nối từ biến môi trường nếu không được cung cấp
            if neo4j_uri is None:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            if neo4j_user is None:
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            if neo4j_password is None:
                neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j")
                
            # Kết nối Neo4j
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            
            # Kiểm tra kết nối
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run("RETURN 1 as n")
                    for record in result:
                        pass  # Chỉ kiểm tra kết nối
                print(f"Neo4j connection successful to {neo4j_uri}")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def __del__(self):
        """Đóng kết nối Neo4j khi xoá đối tượng"""
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()
    
    def extract_subgraph(self, 
                       seed_nodes: List[int],
                       alpha: float = 0.8,
                       max_nodes: int = 6000,
                       max_iterations: int = 20,
                       min_prob: float = 1e-5,
                       label_filters: Optional[List[str]] = None) -> Tuple[List[int], List[Tuple[int, str, int]]]:
        """
        Trích xuất tiểu đồ thị liên quan nhất bằng MixPR
        
        Args:
            seed_nodes: Danh sách các nút hạt giống ban đầu
            alpha: Hệ số random walk (alpha cao = khám phá rộng hơn)
            max_nodes: Kích thước tối đa của tiểu đồ thị (số lượng nút)
            max_iterations: Số lần lặp tối đa PageRank
            min_prob: Xác suất tối thiểu để đưa vào tiểu đồ thị
            label_filters: Danh sách các nhãn nút cần lọc (chỉ dùng với Neo4j)
            
        Returns:
            Tuple(nodes, edges):
                - nodes: Danh sách các nút trong tiểu đồ thị
                - edges: Danh sách các cạnh trong tiểu đồ thị (src, rel, dst)
        """
        if self.use_neo4j:
            return self._extract_subgraph_neo4j(
                seed_nodes, alpha, max_nodes, max_iterations, min_prob, label_filters
            )
        else:
            return self._extract_subgraph_networkx(
                seed_nodes, alpha, max_nodes, max_iterations, min_prob
            )
    
    def _extract_subgraph_networkx(self,
                                 seed_nodes: List[int],
                                 alpha: float,
                                 max_nodes: int,
                                 max_iterations: int,
                                 min_prob: float) -> Tuple[List[int], List[Tuple[int, str, int]]]:
        """
        Thực hiện MixPR với NetworkX
        
        Args:
            Xem extract_subgraph để biết mô tả đầy đủ
            
        Returns:
            Tuple(nodes, edges)
        """
        if self.graph is None:
            raise ValueError("No in-memory graph provided. Create a graph or use Neo4j.")
        
        print(f"Running MixPR using NetworkX with alpha={alpha}, seed_nodes={seed_nodes}")
        
        # Đảm bảo tất cả các nút hạt giống đều có trong đồ thị
        valid_seed_nodes = [node for node in seed_nodes if node in self.graph.nodes()]
        
        if not valid_seed_nodes:
            raise ValueError("None of the seed nodes are in the graph")
            
        if len(valid_seed_nodes) < len(seed_nodes):
            print(f"Warning: Only {len(valid_seed_nodes)}/{len(seed_nodes)} seed nodes found in graph")
            
        # Khởi tạo vector cá nhân hóa (personalization vector)
        personalization = {}
        for node in self.graph.nodes():
            if node in valid_seed_nodes:
                personalization[node] = 1.0 / len(valid_seed_nodes)
            else:
                personalization[node] = 0.0
        
        # Chạy Personalized PageRank
        pagerank = nx.pagerank(
            self.graph,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iterations,
            tol=1e-6
        )
        
        # Lọc các nút có xác suất cao nhất
        important_nodes = sorted(
            [(node, score) for node, score in pagerank.items() if score >= min_prob],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Lấy max_nodes nút quan trọng nhất
        top_nodes = [node for node, _ in important_nodes[:max_nodes]]
        
        # Trích xuất các cạnh liên quan
        edges = []
        for src in top_nodes:
            for dst in self.graph.successors(src):
                if dst in top_nodes:
                    # Lấy các thuộc tính của cạnh (nếu có)
                    edge_data = self.graph.get_edge_data(src, dst)
                    rel_type = edge_data.get('type', 'related_to') if edge_data else 'related_to'
                    edges.append((src, rel_type, dst))
        
        print(f"Extracted subgraph with {len(top_nodes)} nodes and {len(edges)} edges")
        
        return top_nodes, edges
    
    def _extract_subgraph_neo4j(self,
                               seed_nodes: List[int],
                               alpha: float,
                               max_nodes: int,
                               max_iterations: int,
                               min_prob: float,
                               label_filters: Optional[List[str]] = None) -> Tuple[List[int], List[Tuple[int, str, int]]]:
        """
        Thực hiện MixPR với Neo4j
        
        Args:
            Xem extract_subgraph để biết mô tả đầy đủ
            
        Returns:
            Tuple(nodes, edges)
        """
        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not initialized")
            
        print(f"Running MixPR using Neo4j with alpha={alpha}, seed_nodes={seed_nodes}")
        
        # Xây dựng truy vấn Cypher cho MixPR
        # Sử dụng APOC PageRank hoặc Graph Algorithms nếu có
        try:
            # Kiểm tra xem APOC có khả dụng không
            with self.neo4j_driver.session() as session:
                result = session.run("CALL dbms.procedures() YIELD name WHERE name STARTS WITH 'apoc' RETURN count(*) as count")
                has_apoc = result.single()["count"] > 0
                
                if has_apoc:
                    nodes, edges = self._run_neo4j_apoc_pagerank(
                        seed_nodes, alpha, max_nodes, max_iterations, min_prob, label_filters
                    )
                else:
                    # Fallback to custom implementation if APOC not available
                    print("APOC not available, using custom implementation")
                    nodes, edges = self._run_neo4j_custom_pagerank(
                        seed_nodes, alpha, max_nodes, max_iterations, min_prob, label_filters
                    )
        except Exception as e:
            print(f"Error using Neo4j PageRank: {e}")
            # Fallback to custom implementation
            nodes, edges = self._run_neo4j_custom_pagerank(
                seed_nodes, alpha, max_nodes, max_iterations, min_prob, label_filters
            )
        
        return nodes, edges
    
    def _run_neo4j_apoc_pagerank(self,
                               seed_nodes: List[int],
                               alpha: float,
                               max_nodes: int,
                               max_iterations: int,
                               min_prob: float,
                               label_filters: Optional[List[str]] = None) -> Tuple[List[int], List[Tuple[int, str, int]]]:
        """MixPR sử dụng APOC PageRank extension của Neo4j"""
        
        # Chuẩn bị danh sách các nút hạt giống với định dạng Cypher
        seed_nodes_str = ", ".join([str(node) for node in seed_nodes])
        
        # Xây dựng điều kiện lọc nhãn (nếu cần)
        label_condition = ""
        if label_filters:
            label_clauses = []
            for label in label_filters:
                label_clauses.append(f"n:{label}")
            label_condition = " WHERE " + " OR ".join(label_clauses)
        
        # Truy vấn Cypher để chạy PageRank và lấy tiểu đồ thị
        # Lấy danh sách nút và cạnh
        query = f"""
        // Step 1: Create a subgraph view (all nodes and edges)
        CALL apoc.graph.fromCypher(
            "MATCH (n){label_condition} RETURN id(n) as id",
            "MATCH (n)-[r]->(m){label_condition} RETURN id(n) as source, id(m) as target, type(r) as type",
            {{}}
        ) YIELD graph as subgraph
        
        // Step 2: Run PageRank on the subgraph with personalization
        CALL apoc.algo.pageRank(subgraph.nodes, subgraph.relationships, 
            {{iterations: {max_iterations}, dampingFactor: {alpha}}}) 
        YIELD node, score
        
        // Step 3: Filter and sort by score
        WITH node, score
        WHERE id(node) IN [{seed_nodes_str}] OR score >= {min_prob}
        ORDER BY score DESC
        LIMIT {max_nodes}
        
        WITH collect(node) as top_nodes
        
        // Step 4: Get the internal edges between these nodes
        MATCH (n)-[r]->(m)
        WHERE n IN top_nodes AND m IN top_nodes
        
        // Step 5: Return the subgraph
        RETURN collect(distinct id(n)) as nodes, 
               collect({{source: id(n), rel_type: type(r), target: id(m)}}) as edges
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            if not record:
                return [], []
                
            nodes = record["nodes"]
            edge_records = record["edges"]
            
            # Chuyển đổi edge records thành danh sách cạnh
            edges = []
            for edge in edge_records:
                edges.append((edge["source"], edge["rel_type"], edge["target"]))
        
        print(f"Neo4j APOC: Extracted subgraph with {len(nodes)} nodes and {len(edges)} edges")
        
        return nodes, edges
    
    def _run_neo4j_custom_pagerank(self,
                                 seed_nodes: List[int],
                                 alpha: float,
                                 max_nodes: int,
                                 max_iterations: int,
                                 min_prob: float,
                                 label_filters: Optional[List[str]] = None) -> Tuple[List[int], List[Tuple[int, str, int]]]:
        """
        MixPR cho Neo4j mà không phụ thuộc vào APOC
        Triển khai PPR tùy chỉnh bằng nhiều truy vấn Cypher
        """
        with self.neo4j_driver.session() as session:
            # 1. Lấy danh sách các nút trong đồ thị
            label_filter = ""
            if label_filters:
                label_filter = " WHERE " + " OR ".join([f"n:{label}" for label in label_filters])
                
            node_query = f"MATCH (n){label_filter} RETURN id(n) as id"
            result = session.run(node_query)
            all_nodes = [record["id"] for record in result]
            
            if not all_nodes:
                raise ValueError("No nodes found in graph")
                
            # 2. Lấy ma trận kề
            edge_query = f"""
            MATCH (n)-[r]->(m){label_filter}
            RETURN id(n) as source, id(m) as target, type(r) as rel_type
            """
            result = session.run(edge_query)
            
            # Tạo ma trận kề dạng sparse
            outlinks = defaultdict(list)
            edge_info = {}  # Lưu thông tin về cạnh
            
            for record in result:
                src = record["source"]
                dst = record["target"]
                rel_type = record["rel_type"]
                
                outlinks[src].append(dst)
                edge_info[(src, dst)] = rel_type
            
            # 3. Tính toán số liên kết ra của mỗi nút
            outlink_counts = {node: len(outlinks[node]) for node in all_nodes}
            
            # 4. Personalization vector
            personalization = {}
            for node in all_nodes:
                if node in seed_nodes:
                    personalization[node] = 1.0 / len(seed_nodes)
                else:
                    personalization[node] = 0.0
            
            # 5. Chạy thuật toán PageRank cá nhân hóa
            # Khởi tạo vector PageRank
            pr = {node: 1.0 / len(all_nodes) for node in all_nodes}
            
            # Lặp PageRank
            for _ in range(max_iterations):
                next_pr = {node: 0.0 for node in all_nodes}
                
                # Phân phối PageRank qua các cạnh
                for node in all_nodes:
                    if outlink_counts[node] > 0:
                        # Chuyển PageRank cho các nút kề
                        share = alpha * pr[node] / outlink_counts[node]
                        for neighbor in outlinks[node]:
                            next_pr[neighbor] += share
                    else:
                        # Nút không có đỉnh đi ra, phân phối đều
                        share = alpha * pr[node] / len(all_nodes)
                        for n in all_nodes:
                            next_pr[n] += share
                
                # Teleportation
                for node in all_nodes:
                    next_pr[node] += (1 - alpha) * personalization[node]
                
                # Cập nhật vector PageRank
                pr = next_pr
            
            # 6. Lọc và sắp xếp các nút
            ranked_nodes = sorted([(node, score) for node, score in pr.items()], 
                                 key=lambda x: x[1], reverse=True)
            
            # Chỉ giữ các nút với điểm cao nhất
            filtered_nodes = [node for node, score in ranked_nodes 
                             if node in seed_nodes or score >= min_prob]
            
            # Lấy tối đa max_nodes
            top_nodes = filtered_nodes[:max_nodes]
            
            # 7. Trích xuất các cạnh trong tiểu đồ thị
            edges = []
            for src in top_nodes:
                for dst in outlinks[src]:
                    if dst in top_nodes:
                        rel_type = edge_info[(src, dst)]
                        edges.append((src, rel_type, dst))
        
        print(f"Neo4j Custom: Extracted subgraph with {len(top_nodes)} nodes and {len(edges)} edges")
        
        return top_nodes, edges
    
    def create_networkx_graph(self, nodes: List[int], edges: List[Tuple[int, str, int]]) -> nx.DiGraph:
        """
        Tạo đồ thị NetworkX từ danh sách nút và cạnh
        
        Args:
            nodes: Danh sách các ID nút
            edges: Danh sách các cạnh (src, rel_type, dst)
            
        Returns:
            Đồ thị NetworkX
        """
        G = nx.DiGraph()
        
        # Thêm các nút
        for node_id in nodes:
            G.add_node(node_id)
        
        # Thêm các cạnh
        for src, rel_type, dst in edges:
            G.add_edge(src, dst, type=rel_type)
        
        return G
    
    def load_graph_from_neo4j(self, 
                            node_labels: Optional[List[str]] = None, 
                            rel_types: Optional[List[str]] = None,
                            limit: Optional[int] = None) -> nx.DiGraph:
        """
        Tải đồ thị từ Neo4j vào bộ nhớ dưới dạng đồ thị NetworkX
        
        Args:
            node_labels: Danh sách các nhãn nút cần tải (None = tất cả)
            rel_types: Danh sách các loại cạnh cần tải (None = tất cả)
            limit: Giới hạn số lượng nút tải (None = không giới hạn)
            
        Returns:
            Đồ thị NetworkX
        """
        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not initialized")
        
        # Xây dựng điều kiện WHERE cho truy vấn nút
        node_condition = ""
        if node_labels:
            label_clauses = []
            for label in node_labels:
                label_clauses.append(f"n:{label}")
            node_condition = " WHERE " + " OR ".join(label_clauses)
        
        # Xây dựng điều kiện WHERE cho truy vấn cạnh
        rel_condition = ""
        if rel_types:
            rel_clauses = []
            for rel_type in rel_types:
                rel_clauses.append(f"type(r) = '{rel_type}'")
            rel_condition = " AND (" + " OR ".join(rel_clauses) + ")"
        
        # Xây dựng giới hạn
        limit_clause = ""
        if limit:
            limit_clause = f" LIMIT {limit}"
        
        # Truy vấn lấy thông tin nút
        node_query = f"""
        MATCH (n){node_condition}
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        {limit_clause}
        """
        
        # Truy vấn lấy thông tin cạnh
        edge_query = f"""
        MATCH (n)-[r]->(m)
        WHERE id(n) IN $node_ids AND id(m) IN $node_ids{rel_condition}
        RETURN id(n) as source, id(m) as target, type(r) as type, properties(r) as properties
        """
        
        G = nx.DiGraph()
        
        with self.neo4j_driver.session() as session:
            # Tải thông tin nút
            result = session.run(node_query)
            node_ids = []
            
            for record in result:
                node_id = record["id"]
                node_ids.append(node_id)
                
                # Thêm nút với các thuộc tính
                G.add_node(node_id, labels=record["labels"], **record["properties"])
            
            # Nếu không có nút, trả về đồ thị rỗng
            if not node_ids:
                return G
                
            # Tải thông tin cạnh
            result = session.run(edge_query, node_ids=node_ids)
            
            for record in result:
                # Thêm cạnh với các thuộc tính
                G.add_edge(
                    record["source"], 
                    record["target"], 
                    type=record["type"], 
                    **record["properties"]
                )
        
        print(f"Loaded graph from Neo4j with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G

def get_subgraph_with_mixpr(seed_nodes: List[int],
                          alpha: float = 0.8,
                          topk: int = 6000,
                          max_steps: int = 20,
                          min_prob: float = 1e-4) -> Tuple[List[int], List[Tuple[int, str, int]]]:
    """
    Trích xuất tiểu đồ thị liên quan đến truy vấn bằng thuật toán MixPR
    
    Args:
        seed_nodes: Danh sách các nút hạt giống
        alpha: Hệ số teleport trong thuật toán PPR
        topk: Kích thước tối đa của tiểu đồ thị (số lượng nút)
        max_steps: Số bước lặp tối đa
        min_prob: Xác suất tối thiểu để xét đến nút
        
    Returns:
        Tuple(nodes, edges):
            - nodes: Danh sách các nút trong tiểu đồ thị
            - edges: Danh sách các cạnh trong tiểu đồ thị
    """
    print(f"Running MixPR with alpha={alpha}, topk={topk}, seed_nodes={seed_nodes}")
    
    # Kiểm tra xem có khả dụng Neo4j không
    use_neo4j = False
    neo4j_uri = os.getenv("NEO4J_URI")
    
    if neo4j_uri and HAS_NEO4J:
        try:
            # Tạo MixPR Router với Neo4j
            router = MixPRRouter(
                use_neo4j=True,
                neo4j_uri=neo4j_uri,
                neo4j_user=os.getenv("NEO4J_USER"),
                neo4j_password=os.getenv("NEO4J_PASSWORD")
            )
            use_neo4j = True
        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")
            print("Falling back to in-memory graph")
            use_neo4j = False
    
    if not use_neo4j:
        # Fallback: Tạo đồ thị mẫu trong bộ nhớ
        sample_graph = _create_sample_graph(seed_nodes)
        
        # Tạo MixPR Router với đồ thị NetworkX
        router = MixPRRouter(
            use_neo4j=False,
            in_memory_graph=sample_graph
        )
    
    # Thực hiện thuật toán MixPR
    return router.extract_subgraph(
        seed_nodes=seed_nodes,
        alpha=alpha,
        max_nodes=topk,
        max_iterations=max_steps,
        min_prob=min_prob
    )

def _create_sample_graph(seed_nodes: List[int]) -> nx.DiGraph:
    """
    Tạo đồ thị mẫu cho demo khi không có Neo4j
    
    Args:
        seed_nodes: Danh sách các ID nút hạt giống
        
    Returns:
        Đồ thị NetworkX
    """
    # Tạo đồ thị có hướng
    G = nx.DiGraph()
    
    # Thêm nút ước lượng 10000 nút
    for i in range(1, 10001):
        G.add_node(i)
    
    # Thêm các nút đặc biệt nếu chưa có
    for node_id in seed_nodes:
        if node_id not in G:
            G.add_node(node_id)
    
    # Thêm các nút đặc biệt (thuốc, bệnh, etc.)
    special_nodes = [
        # Drug nodes (ID: 1000-1099)
        1001, 1002, 1003, 1004, 1005,
        # Disease nodes (ID: 1100-1199)
        1101, 1102, 1103,
        # Mechanism nodes (ID: 1200-1299)
        1201, 1202, 1203,
        # Anatomy nodes (ID: 1300-1399)
        1301, 1302
    ]
    
    for node_id in special_nodes:
        if node_id not in G:
            G.add_node(node_id)
    
    # Thêm các cạnh đặc biệt
    special_edges = [
        # Drug-Disease edges
        (1001, "treats", 1101),
        (1002, "treats", 1101),
        (1003, "treats", 1101),
        (1004, "treats", 1101),
        (1005, "treats", 1101),
        
        # Drug-Mechanism edges
        (1001, "inhibits", 1201),
        (1002, "inhibits", 1203),
        (1003, "stimulates", 1202),
        (1004, "inhibits", 1201),
        (1004, "stimulates", 1202),
        (1005, "inhibits", 1201),
        
        # Disease-Disease edges
        (1102, "progresses_to", 1101),
        (1101, "causes", 1103),
        
        # Mechanism-Mechanism edges
        (1203, "regulates", 1201),
        
        # Anatomy edges
        (1301, "part_of", 1302),
    ]
    
    for src, rel, dst in special_edges:
        G.add_edge(src, dst, type=rel)
    
    # Thêm các cạnh ngẫu nhiên để mô phỏng đồ thị dày đặc
    np.random.seed(42)  # Để kết quả nhất quán
    
    # Với mỗi nút, thêm một số cạnh ngẫu nhiên
    for node in G.nodes():
        # Số cạnh ra tỷ lệ với ID nút (chia 1000) nhưng tối đa là 10
        num_edges = min(10, 1 + int(node / 1000))
        
        for _ in range(num_edges):
            # Chọn ngẫu nhiên nút đích
            target = np.random.randint(1, 10001)
            
            # Chọn loại quan hệ
            rel_types = ["related_to", "associated_with", "co_occurs_with", "has_part", "references"]
            rel = np.random.choice(rel_types)
            
            # Thêm cạnh
            G.add_edge(node, target, type=rel)
    
    print(f"Created sample graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G 