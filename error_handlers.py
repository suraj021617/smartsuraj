"""
Error handlers for Flask application
"""
from flask import render_template, jsonify, request
import logging

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Not found'}), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f'Server Error: {error}')
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Forbidden'}), 403
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f'Unhandled Exception: {error}', exc_info=True)
        if request.path.startswith('/api/'):
            return jsonify({'error': 'An error occurred'}), 500
        return render_template('errors/500.html'), 500
