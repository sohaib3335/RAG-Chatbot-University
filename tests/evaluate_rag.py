"""
RAG System Evaluation Module
Evaluates the RAG chatbot against test queries and expected responses
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_chain import RAGChain


class RAGEvaluator:
    """
    Evaluator class for testing RAG system performance
    """
    
    def __init__(self, rag_chain: RAGChain = None):
        """
        Initialize evaluator
        
        Args:
            rag_chain: RAGChain instance to evaluate
        """
        self.rag_chain = rag_chain
        self.test_queries = []
        self.expected_responses = []
        self.results = []
        
    def load_test_data(
        self,
        queries_path: str = None,
        expected_path: str = None
    ):
        """
        Load test queries and expected responses
        
        Args:
            queries_path: Path to test queries JSON file
            expected_path: Path to expected responses JSON file
        """
        base_path = Path(__file__).parent
        
        queries_path = queries_path or base_path / "test_queries.json"
        expected_path = expected_path or base_path / "expected_responses.json"
        
        with open(queries_path, 'r') as f:
            data = json.load(f)
            self.test_queries = data.get('test_queries', [])
        
        with open(expected_path, 'r') as f:
            data = json.load(f)
            self.expected_responses = {
                r['query_id']: r 
                for r in data.get('expected_responses', [])
            }
        
        print(f"Loaded {len(self.test_queries)} test queries")
        print(f"Loaded {len(self.expected_responses)} expected responses")
    
    def evaluate_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            query_data: Query data dictionary
            
        Returns:
            Evaluation result dictionary
        """
        query_id = query_data['id']
        query = query_data['query']
        expected = self.expected_responses.get(query_id, {})
        
        # Get response from RAG system
        result = self.rag_chain.query(query)
        answer = result.get('answer', '')
        
        # Evaluate containment
        expected_contains = expected.get('expected_answer_contains', [])
        matches = []
        missing = []
        
        for term in expected_contains:
            if term.lower() in answer.lower():
                matches.append(term)
            else:
                missing.append(term)
        
        # Calculate score
        if expected_contains:
            score = len(matches) / len(expected_contains)
        else:
            score = 0.0
        
        return {
            'query_id': query_id,
            'query': query,
            'category': query_data.get('category', 'unknown'),
            'difficulty': query_data.get('difficulty', 'unknown'),
            'answer': answer,
            'expected_source': expected.get('expected_source', ''),
            'expected_terms': expected_contains,
            'matched_terms': matches,
            'missing_terms': missing,
            'score': score,
            'num_sources': result.get('num_retrieved', 0),
            'passed': score >= 0.5  # Pass if at least 50% of expected terms are present
        }
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all test queries
        
        Returns:
            Summary of evaluation results
        """
        if not self.rag_chain or not self.rag_chain.is_initialized:
            print("Error: RAG chain not initialized")
            return {"error": "RAG chain not initialized"}
        
        self.results = []
        
        print("\n" + "="*60)
        print("RAG SYSTEM EVALUATION")
        print("="*60 + "\n")
        
        for i, query_data in enumerate(self.test_queries, 1):
            print(f"[{i}/{len(self.test_queries)}] Evaluating: {query_data['query'][:50]}...")
            
            try:
                result = self.evaluate_single_query(query_data)
                self.results.append(result)
                
                status = "✓ PASS" if result['passed'] else "✗ FAIL"
                print(f"  {status} (Score: {result['score']:.2f})")
                
            except Exception as e:
                print(f"  ✗ ERROR: {str(e)}")
                self.results.append({
                    'query_id': query_data['id'],
                    'query': query_data['query'],
                    'error': str(e),
                    'passed': False,
                    'score': 0.0
                })
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate evaluation summary
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Calculate metrics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passed', False))
        scores = [r['score'] for r in self.results if 'score' in r]
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Breakdown by category
        categories = {}
        for r in self.results:
            cat = r.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0, 'scores': []}
            categories[cat]['total'] += 1
            if r.get('passed', False):
                categories[cat]['passed'] += 1
            categories[cat]['scores'].append(r.get('score', 0))
        
        for cat in categories:
            categories[cat]['avg_score'] = (
                sum(categories[cat]['scores']) / len(categories[cat]['scores'])
                if categories[cat]['scores'] else 0
            )
            del categories[cat]['scores']  # Remove raw scores from summary
        
        # Breakdown by difficulty
        difficulties = {}
        for r in self.results:
            diff = r.get('difficulty', 'unknown')
            if diff not in difficulties:
                difficulties[diff] = {'total': 0, 'passed': 0}
            difficulties[diff]['total'] += 1
            if r.get('passed', False):
                difficulties[diff]['passed'] += 1
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_score': avg_score,
            'by_category': categories,
            'by_difficulty': difficulties
        }
        
        return summary
    
    def save_results(self, output_path: str = None):
        """
        Save evaluation results to file
        
        Args:
            output_path: Path to save results
        """
        output_path = output_path or Path(__file__).parent / "evaluation_results.json"
        
        summary = self.generate_summary()
        
        output = {
            'summary': summary,
            'detailed_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_report(self):
        """Print a formatted evaluation report"""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Queries: {summary['total_queries']}")
        print(f"   Passed: {summary['passed']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Pass Rate: {summary['pass_rate']:.1%}")
        print(f"   Average Score: {summary['average_score']:.2f}")
        
        print(f"\n📁 By Category:")
        for cat, data in summary['by_category'].items():
            print(f"   {cat}: {data['passed']}/{data['total']} (avg: {data['avg_score']:.2f})")
        
        print(f"\n📈 By Difficulty:")
        for diff, data in summary['by_difficulty'].items():
            rate = data['passed'] / data['total'] if data['total'] > 0 else 0
            print(f"   {diff}: {data['passed']}/{data['total']} ({rate:.1%})")
        
        print("\n" + "="*60)


def run_evaluation(use_local: bool = True):
    """
    Run full evaluation
    
    Args:
        use_local: Whether to use local models
    """
    # Initialize RAG chain
    print("Initializing RAG system...")
    rag_chain = RAGChain(
        use_local_embeddings=use_local,
        use_local_llm=use_local
    )
    
    # Load existing store
    if not rag_chain.load_existing_store():
        print("No indexed documents found. Please run document ingestion first.")
        print("Run: python main.py ingest")
        return
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_chain)
    evaluator.load_test_data()
    
    # Run evaluation
    evaluator.evaluate_all()
    evaluator.print_report()
    evaluator.save_results()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument('--local', '-l', action='store_true', help='Use local models')
    args = parser.parse_args()
    
    run_evaluation(use_local=args.local)
