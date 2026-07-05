# backend/api.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime
import json
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


def _format_path(path_dict):
    return {
        "nodes": path_dict.get('nodes', []),
        "node_types": path_dict.get('node_types', []),
        "relationships": path_dict.get('relationships', []),
        "score": path_dict.get('score', 0.0),
        "evidence": path_dict.get('evidence', [])
    }


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
            path_dict = path.to_dict() if hasattr(path, 'to_dict') else path
            paths_response.append(_format_path(path_dict))

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


# Streaming query endpoint - pushes beam search progress as it happens,
# instead of waiting for the whole search to finish. Local dev only: relies
# on a long-lived connection, which Vercel's serverless functions don't support.
@app.route("/query/stream", methods=["POST"])
def query_knowledge_graph_stream():
    global reasoner

    if not reasoner:
        return jsonify({
            "error": "Reasoner not initialized"
        }), 503

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

    beam_width = data.get("beam_width", reasoner.beam_width)
    max_depth = data.get("max_depth", reasoner.max_depth)

    if beam_width != reasoner.beam_width or max_depth != reasoner.max_depth:
        reasoner.close()
        reasoner = BeamSearchReasoner(beam_width=beam_width, max_depth=max_depth)

    def generate():
        try:
            for event in reasoner.answer_question_stream(query):
                if event.get("type") in ("depth", "final") and "paths" in event:
                    event = {**event, "paths": [_format_path(p) for p in event["paths"]]}
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": f"Query failed: {str(e)}"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


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


# Expand a single node's neighbors, for click-to-expand graph exploration
@app.route("/neighbors", methods=["GET"])
def get_node_neighbors():
    global reasoner

    if not reasoner:
        return jsonify({
            "error": "Reasoner not initialized"
        }), 503

    name = request.args.get("name", "").strip()
    node_type = request.args.get("type", "").strip()

    if not name or not node_type:
        return jsonify({
            "error": "Both 'name' and 'type' query parameters are required"
        }), 400

    try:
        neighbors = reasoner.get_neighbors(name, node_type)

        nodes = [
            {"id": n["name"], "name": n["name"], "type": n["type"]}
            for n in neighbors
        ]

        links = []
        for n in neighbors:
            if n.get("direction", "outgoing") == "outgoing":
                links.append({"source": name, "target": n["name"], "relationship": n["relationship"]})
            else:
                links.append({"source": n["name"], "target": name, "relationship": n["relationship"]})

        return jsonify({"nodes": nodes, "links": links})

    except Exception as e:
        return jsonify({
            "error": f"Neighbor query failed: {str(e)}"
        }), 500


# Full graph, for the "view entire graph" button. The KG is small (~60 drugs,
# a couple dozen conditions) so fetching everything in one shot is cheap.
@app.route("/graph", methods=["GET"])
def get_full_graph():
    global reasoner

    if not reasoner:
        return jsonify({
            "error": "Reasoner not initialized"
        }), 503

    try:
        with reasoner.driver.session(database=reasoner.database) as session:
            nodes_result = session.run(
                "MATCH (n) RETURN n.name AS name, labels(n)[0] AS type"
            )
            nodes = [
                {"id": record["name"], "name": record["name"], "type": record["type"]}
                for record in nodes_result
            ]

            links_result = session.run(
                "MATCH (a)-[r]->(b) RETURN a.name AS source, b.name AS target, type(r) AS relationship"
            )
            links = [
                {
                    "source": record["source"],
                    "target": record["target"],
                    "relationship": record["relationship"],
                }
                for record in links_result
            ]

        return jsonify({"nodes": nodes, "links": links})

    except Exception as e:
        return jsonify({
            "error": f"Graph query failed: {str(e)}"
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
