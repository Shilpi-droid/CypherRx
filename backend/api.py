# backend/api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.beam_searcher import BeamSearchReasoner

app = Flask(__name__)

# CORS middleware
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],  # React dev server
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize reasoner globally
reasoner = None


# Initialize reasoner immediately
try:
    beam_width = int(os.getenv("BEAM_WIDTH", 5))
    max_depth = int(os.getenv("MAX_DEPTH", 4))
    reasoner = BeamSearchReasoner(beam_width=beam_width, max_depth=max_depth)
    print("✓ Reasoner initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize reasoner: {e}")
    reasoner = None


# Cleanup on app shutdown
@app.teardown_appcontext
def close_reasoner(error):
    global reasoner
    if reasoner:
        # Don't close here - keep it alive for multiple requests
        pass


# Health check
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "reasoner_initialized": reasoner is not None
    })


# Main query endpoint
@app.route("/query", methods=["POST"])
def query_knowledge_graph():
    global reasoner
    
    if not reasoner:
        return jsonify({
            "error": "Reasoner not initialized"
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        query = data.get("query", "").strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Optionally override beam_width and max_depth from request
        beam_width = data.get("beam_width", reasoner.beam_width)
        max_depth = data.get("max_depth", reasoner.max_depth)
        
        # Update reasoner parameters if different
        if beam_width != reasoner.beam_width or max_depth != reasoner.max_depth:
            reasoner.close() if reasoner else None
            reasoner = BeamSearchReasoner(beam_width=beam_width, max_depth=max_depth)
        
        # Execute query
        result = reasoner.answer_question(query)
        
        # Convert paths to response format
        paths_response = []
        for path in result.get('paths', [])[:3]:  # Return top 3 paths
            # Handle both dict and Path object
            if hasattr(path, 'to_dict'):
                path_dict = path.to_dict()
            else:
                path_dict = path
            
            paths_response.append({
                "nodes": path_dict.get('nodes', []),
                "node_types": path_dict.get('node_types', []),
                "relationships": path_dict.get('relationships', []),
                "score": path_dict.get('score', 0.0),
                "evidence": path_dict.get('evidence', [])
            })
        
        response = {
            "answer": result.get('answer', ''),
            "paths": paths_response,
            "confidence": result.get('confidence', 0.0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": f"Query failed: {str(e)}"
        }), 500


# Get graph statistics
@app.route("/stats", methods=["GET"])
def get_graph_stats():
    global reasoner
    
    if not reasoner:
        return jsonify({
            "error": "Reasoner not initialized"
        }), 503
    
    try:
        with reasoner.driver.session(database=reasoner.database) as session:
            # Count nodes by type
            drugs_result = session.run("MATCH (d:Drug) RETURN count(d) as count")
            drugs_count = drugs_result.single()['count']
            
            diseases_result = session.run("MATCH (d:Disease) RETURN count(d) as count")
            diseases_count = diseases_result.single()['count']
            
            side_effects_result = session.run("MATCH (s:SideEffect) RETURN count(s) as count")
            side_effects_count = side_effects_result.single()['count']
            
            # Count relationships
            rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rels_count = rels_result.single()['count']
            
            return jsonify({
                "total_drugs": drugs_count,
                "total_diseases": diseases_count,
                "total_side_effects": side_effects_count,
                "total_relationships": rels_count
            })
    
    except Exception as e:
        return jsonify({
            "error": f"Stats query failed: {str(e)}"
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    # Run the Flask app
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    print(f"Starting Flask API server on port {port}")
    print(f"Debug mode: {debug}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
