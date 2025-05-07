import logging
import os
import json
import numpy as np
import pandas as pd
from functools import wraps
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import time
import asyncio
from io import StringIO
import tempfile
from openai import AzureOpenAI
from fuzzywuzzy import process
_positive_anchor_embedding = None
_negative_anchor_embedding = None
def async_timeout(seconds):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logging.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail=f"Request timed out after {seconds} seconds"
                )
        return wrapper
    return decorator
# Configure detailed logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("customer_feedback.log")
    ]
)
logger = logging.getLogger("customer_feedback")

# Azure OpenAI client configuration - reuse from main app
AZURE_ENDPOINT = "https://prodhubfinnew-openai-97de.openai.azure.com/"
AZURE_API_KEY = "97fa8c02f9e64e8ea5434987b11fe6f4" 
AZURE_API_VERSION = "2024-12-01-preview"  # Updated API version

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # Correct embedding model name
AZURE_CLIENT = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY
)


# Create standalone app (primary usage mode)
app = FastAPI(
    title="Customer Feedback Analysis API",
    description="API for analyzing customer feedback and identifying key problem areas",
    version="1.0.0"
)

# Add CORS middleware to allow cross-domain requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=120)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timed out after 120 seconds"}
        )
# Router is kept for optional integration scenarios
router = APIRouter()

# Common key areas from screenshot and expanded options
# Change from dictionary format to simple list of areas
COMMON_KEY_AREAS = [
    "Reporting & Analytics",
    "Automation & Workflows",
    "Mobile Experience",
    "Customization & Configurability",
    "Integrations & API Access",
    "Access Control & Permissions",
    "Collaboration & Multi-Team Support",
    "AI-Powered Decision Support",
    "User Interface",
    "Performance & Speed",
    "Data Visualization",
    "Onboarding & Training",
    "Customer Support",
    "Pricing & Cost",
    "Feature Requests",
    "Bugs & Stability",
    "Documentation",
    "Reliability",
    "Search Functionality",
    "Data Management",
    "Security & Privacy",
    "Cross-platform Compatibility",
    "Accessibility",
    "Notifications & Alerts",
    "User Roles",
    "App Navigation",
    "App Crashes & Stability",
    "Login & Authentication",
    "App Response Time",
    "Checkout Process",
    "Mobile App Design",
    "Cart Functionality",
    "App Update Issues",
    "Order Tracking",
    "Payment Options",
    "Account Management",
    "App Search Functionality",
    "Push Notifications",
    "Wishlist & Favorites",
    "App Filters & Sorting"
]
def standardize_dataframe(df: pd.DataFrame, columns: Dict[str, str], source: str = None) -> pd.DataFrame:
    """
    Standardizes a dataframe to have consistent column names and structure.
    
    Args:
        df: Original DataFrame
        columns: Dictionary mapping column types to actual column names
        source: Optional source value to override the Source column
        
    Returns:
        Standardized DataFrame with consistent columns
    """
    try:
        logger.info("Standardizing DataFrame to consistent column structure")
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create a mapping from original column names to standard names
        column_mapping = {}
        if columns["feedback_col"]:
            column_mapping[columns["feedback_col"]] = "Customer Feedback"
        if columns["received_col"]:
            column_mapping[columns["received_col"]] = "Received"
        if columns["name_col"]:
            column_mapping[columns["name_col"]] = "Name"
        if columns["source_col"]:
            column_mapping[columns["source_col"]] = "Source"
            
        # Rename the columns that were identified
        result_df = result_df.rename(columns=column_mapping)
        
        # Get current columns and desired columns
        current_columns = set(result_df.columns)
        desired_columns = {"Received", "Name", "Customer Feedback", "Source"}
        
        # Add any missing desired columns with N/A values
        for col in desired_columns:
            if col not in current_columns:
                logger.info(f"Adding missing column '{col}' with N/A values")
                result_df[col] = "N/A"
        
        # Override Source column if source parameter is provided
        if source:
            logger.info(f"Overriding Source column with value: {source}")
            result_df["Source"] = source
                
        # Keep only the desired columns plus any that should be preserved
        columns_to_keep = list(desired_columns)
        
        # Add the rating column if it exists and is different from other columns
        if columns["rating_col"] and columns["rating_col"] not in column_mapping:
            columns_to_keep.append(columns["rating_col"])
        
        # Keep only the desired columns
        result_df = result_df[columns_to_keep]
        
        logger.info(f"Standardized DataFrame structure: {list(result_df.columns)}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error standardizing DataFrame: {str(e)}")
        # Return original dataframe if anything goes wrong
        return df
async def get_sentiment_anchor_embeddings(client: AzureOpenAI) -> Tuple[List[float], List[float]]:
    """
    Generates and caches embeddings for positive and negative sentiment anchors.
    
    Args:
        client: Azure OpenAI client
        
    Returns:
        Tuple of (positive_anchor_embedding, negative_anchor_embedding)
    """
    global _positive_anchor_embedding, _negative_anchor_embedding
    
    # Return cached embeddings if already generated
    if _positive_anchor_embedding is not None and _negative_anchor_embedding is not None:
        return _positive_anchor_embedding, _negative_anchor_embedding
    
    # Define positive and negative sentiment anchor texts
    positive_anchors = [
    "excellent",
    "amazing",
    "fantastic",
    "great",
    "perfect",
    "love it",
    "impressive",
    "satisfied",
    "wonderful",
    "delightful"
    ]
    
    # Better negative anchors (sentiment-pure)
    negative_anchors = [
        "terrible",
        "awful",
        "horrible",
        "disappointing",
        "frustrating",
        "useless",
        "hate it",
        "poor",
        "annoying",
        "dreadful"
    ]
        
    try:
        # Generate embeddings for all anchor texts
        all_anchors = positive_anchors + negative_anchors
        anchor_embeddings = await get_embeddings(client, all_anchors)
        
        # Calculate average embeddings for positive and negative anchors
        positive_embeddings = anchor_embeddings[:len(positive_anchors)]
        negative_embeddings = anchor_embeddings[len(positive_anchors):]
        
        # Calculate average embeddings
        _positive_anchor_embedding = np.mean(positive_embeddings, axis=0).tolist()
        _negative_anchor_embedding = np.mean(negative_embeddings, axis=0).tolist()
        
        logger.info("Generated sentiment anchor embeddings successfully")
        
        return _positive_anchor_embedding, _negative_anchor_embedding
    
    except Exception as e:
        logger.error(f"Error generating sentiment anchor embeddings: {str(e)}")
        
        # Generate fallback embeddings with the same dimension as expected
        dim = 1536  # text-embedding-3-large dimension
        _positive_anchor_embedding = list(np.random.normal(0.5, 0.1, dim))
        _negative_anchor_embedding = list(np.random.normal(-0.5, 0.1, dim))
        
        return _positive_anchor_embedding, _negative_anchor_embedding

async def calculate_sentiment_scores(review_embeddings: List[List[float]], client: AzureOpenAI) -> List[float]:
    """
    Calculates sentiment scores for reviews by comparing their embeddings to positive/negative anchors.
    
    Args:
        review_embeddings: List of review embedding vectors
        client: Azure OpenAI client for generating anchor embeddings if needed
        
    Returns:
        List of sentiment scores between -1 and 1 for each review
    """
    try:
        # Get positive and negative anchor embeddings
        positive_embedding, negative_embedding = await get_sentiment_anchor_embeddings(client)
        
        # Convert to numpy arrays for vectorized operations
        review_matrix = np.array(review_embeddings)
        pos_embedding = np.array(positive_embedding)
        neg_embedding = np.array(negative_embedding)
        
        # Normalize anchor embeddings
        pos_embedding = pos_embedding / np.linalg.norm(pos_embedding)
        neg_embedding = neg_embedding / np.linalg.norm(neg_embedding)
        
        # Normalize review embeddings
        review_norms = np.linalg.norm(review_matrix, axis=1, keepdims=True)
        review_norm = review_matrix / review_norms
        
        # Calculate similarity to positive and negative anchors using vectorized operations
        positive_sim = np.dot(review_norm, pos_embedding)
        negative_sim = np.dot(review_norm, neg_embedding)
        
        # Calculate sentiment scores: scale from -1 to 1
        # Higher positive similarity and lower negative similarity = higher score
        sentiment_scores = positive_sim - negative_sim
        
        # Clip values to ensure they stay in [-1, 1] range (should be already, but just to be safe)
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        logger.info(f"Calculated sentiment scores for {len(review_embeddings)} reviews")
        return sentiment_scores.tolist()
        
    except Exception as e:
        logger.error(f"Error calculating sentiment scores: {str(e)}")
        # Return neutral sentiment scores as fallback
        return [0.0] * len(review_embeddings)
async def process_excel_data(
    client: AzureOpenAI, 
    file_content: bytes,
    filename: str,
    source: str = None  # Add optional source parameter
) -> Dict[str, Any]:
    """
    Process Excel file data to extract and classify feedback from all sheets.
    
    Args:
        client: Azure OpenAI client
        file_content: Raw Excel file content
        filename: Original filename for logging
        source: Optional source value to override the Source column
        
    Returns:
        Dictionary with key areas and classified feedback
    """
    try:
        # Save the content to a temporary file for pandas to read
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
            temp.write(file_content)
            temp_path = temp.name
            logger.info(f"Saved Excel file to temporary path: {temp_path}")
        
        # Read all sheets from the Excel file
        logger.info(f"Reading Excel file with multiple sheets: {filename}")
        excel_file = pd.ExcelFile(temp_path)
        sheet_names = excel_file.sheet_names
        logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        # Process each sheet and collect results
        all_feedback_items = []
        sheet_dfs = {}
        
        for sheet_name in sheet_names:
            logger.info(f"Processing sheet: {sheet_name}")
            try:
                df = pd.read_excel(temp_path, sheet_name=sheet_name)
                if not df.empty:
                    sheet_dfs[sheet_name] = df
                    logger.info(f"Sheet {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Use identify_relevant_columns to find feedback and rating columns
                    columns = await identify_relevant_columns(AZURE_CLIENT, df)
                    feedback_col = columns.get("feedback_col")
                    rating_col = columns.get("rating_col")
                    
                    logger.info(f"Sheet {sheet_name} columns: feedback={feedback_col}, rating={rating_col}")
                    
                    # Standardize the dataframe with the optional source parameter
                    if feedback_col and feedback_col in df.columns:
                        df = standardize_dataframe(df, columns, source)
                        # Update column references after standardization
                        feedback_col = "Customer Feedback"
                        # If rating column was renamed, update the reference
                        if rating_col in columns and columns[rating_col] == feedback_col:
                            rating_col = None
                    
                    # Extract feedback items from this sheet
                    if feedback_col and feedback_col in df.columns:
                        # Skip 5-star reviews if rating column exists
                        if rating_col and rating_col in df.columns:
                            try:
                                df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
                                high_rating_mask = df[rating_col] >= 5
                                five_star_count = high_rating_mask.sum()
                                logger.info(f"Skipping {five_star_count} high-rated reviews (5-star)")
                                df = df[~high_rating_mask]
                            except:
                                logger.warning("Could not filter out 5-star reviews, processing all")
                        
                        # Extract feedback with ratings
                        for idx, row in df.iterrows():
                            if pd.notna(row[feedback_col]) and str(row[feedback_col]).strip():
                                # Include rating if available
                                feedback_text = str(row[feedback_col])
                                if rating_col and rating_col in df.columns and pd.notna(row[rating_col]):
                                    feedback_text = f"[Rating: {row[rating_col]}] {feedback_text}"
                                row_dict = row.replace([float('nan'), float('inf'), float('-inf')], -1).to_dict()
                                all_feedback_items.append({
                                    "text": feedback_text,
                                    "original_index": idx,
                                    "original_row": row_dict,
                                    "sheet_name": sheet_name  # Add sheet name for reference
                                })
                    else:
                        # If no feedback column identified, use the first text column
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                for idx, row in df.iterrows():
                                    if pd.notna(row[col]) and str(row[col]).strip():
                                        all_feedback_items.append({
                                            "text": str(row[col]),
                                            "original_index": idx,
                                            "original_row": row.to_dict(),
                                            "sheet_name": sheet_name
                                        })
                                break
            except Exception as sheet_error:
                logger.error(f"Error processing sheet {sheet_name}: {str(sheet_error)}")
                # Continue with other sheets even if one fails
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
            logger.info(f"Removed temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {temp_path}: {str(e)}")
        
        if not all_feedback_items:
            raise ValueError("No valid feedback found in any sheet of the Excel file")
            
        logger.info(f"Extracted {len(all_feedback_items)} feedback items from {len(sheet_dfs)} sheets")
            
        # Sample for key area identification (limit to 100 items to avoid token limits)
        sample_feedbacks = [item["text"] for item in all_feedback_items[:min(1000,len(all_feedback_items) // 2)]]
        sample_feedback_text = '\n'.join(sample_feedbacks)
        
        # Identify key areas
        key_areas = await identify_key_areas(AZURE_CLIENT, sample_feedback_text)
            
        logger.info(f"Identified {len(key_areas)} key areas")
        
        # IMPORTANT: Do not convert from dict to string here, preserve the full structure
        # Initialize result structure with empty lists - use the area name as key
        classified_feedback = {area.get('area', f"Area {i+1}"): [] for i, area in enumerate(key_areas)}
        
        # Use embeddings for classification
        try:
            logger.info("Using embeddings for feedback classification")
            
            # Create a list of just the area names for embeddings
            problem_texts = [area.get('problem', f"Problem {i+1}") for i, area in enumerate(key_areas)]
            key_area_embeddings = await get_embeddings(AZURE_CLIENT, problem_texts)
            area_names = [area.get('area', f"Area {i+1}") for i, area in enumerate(key_areas)]
            
            # Process feedback in batches for embedding
            feedback_texts = [item["text"] for item in all_feedback_items]
            batch_size = 300
            all_feedback_embeddings = []
            
            for i in range(0, len(feedback_texts), batch_size):
                batch = feedback_texts[i:i + batch_size]
                logger.info(f"Generating embeddings for feedback batch {i//batch_size + 1}/{(len(feedback_texts) + batch_size - 1)//batch_size}")
                batch_embeddings = await get_embeddings(AZURE_CLIENT, batch)
                all_feedback_embeddings.extend(batch_embeddings)
            
            # Calculate sentiment scores using the same embeddings we already generated
            logger.info("Calculating sentiment scores for all feedback items")
            sentiment_scores = await calculate_sentiment_scores(all_feedback_embeddings, AZURE_CLIENT)
            
            # Create a mapping from text to sentiment score
            text_to_sentiment = {feedback_texts[i]: sentiment_scores[i] for i in range(len(feedback_texts))}
            
            # Calculate similarity and classify using numpy
            key_area_matrix = np.array(key_area_embeddings)
            feedback_matrix = np.array(all_feedback_embeddings)
            
            # Verify dimensions match
            key_area_dim = key_area_matrix.shape[1]
            feedback_dim = feedback_matrix.shape[1]
            logger.info(f"Embedding dimensions - Key areas: {key_area_matrix.shape}, Feedback: {feedback_matrix.shape}")
            
            if key_area_dim != feedback_dim:
                logger.error(f"Embedding dimensions mismatch: key_areas({key_area_dim}) != feedback({feedback_dim})")
                raise ValueError(f"Embedding dimensions don't match: {key_area_dim} vs {feedback_dim}")
            
            # Calculate similarity
            def cosine_similarity_matrix(A, B):
                A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
                B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
                return np.dot(A_norm, B_norm.T)
            
            similarity_matrix = cosine_similarity_matrix(feedback_matrix, key_area_matrix)
            
            # Classify feedback based on similarity
            similarity_threshold = 0.7
            
            for i, similarities in enumerate(similarity_matrix):
                # Get indices where similarity exceeds threshold
                matches = np.where(similarities > similarity_threshold)[0]
                
                # If no matches, use the best match
                if len(matches) == 0:
                    best_match = np.argmax(similarities)
                    matches = [best_match]
                
                # Add feedback to all matching areas
                for match_idx in matches:
                    if match_idx < len(area_names):
                        area = area_names[match_idx]
                        classified_feedback[area].append(feedback_texts[i])
            
            logger.info(f"Successfully classified {len(all_feedback_items)} feedback items using embeddings")
            
        except Exception as e:
            logger.error(f"Error with embedding classification: {str(e)}")
            logger.info("Falling back to direct OpenAI classification")
            
            # Fall back to chunked classification via OpenAI
            chunk_size = 10
            for i in range(0, len(all_feedback_items), chunk_size):
                logger.info(f"Processing feedback chunk {i//chunk_size + 1}/{(len(all_feedback_items) + chunk_size - 1)//chunk_size}")
                
                chunk = all_feedback_items[i:i + min(chunk_size, len(all_feedback_items) - i)]
                chunk_texts = [item["text"] for item in chunk]
                
                # Create prompt and classify
                areas_text = "\n".join([f"{j+1}. {area.get('area', f'Area {j+1}')}" for j, area in enumerate(key_areas)])
                feedback_text = "\n".join([f"Feedback {j+1}: {text}" for j, text in enumerate(chunk_texts)])
                
                prompt = f"""
                I need to classify these customer feedback items into the most relevant categories.
                
                CATEGORIES:
                {areas_text}
                
                FEEDBACK ITEMS:
                {feedback_text}
                
                For each feedback item, tell me which categories it belongs to (multiple categories allowed).
                Format as JSON:
                {{
                    "classifications": [
                        {{
                            "feedback_index": 0,
                            "category_indices": [0, 2]
                        }},
                        ...
                    ]
                }}
                
                Category indices are 0-based, corresponding to the numbered list above.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": "You are a customer feedback classification system."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    
                    # Process classifications
                    if "classifications" in result:
                        for classification in result["classifications"]:
                            feedback_idx = classification.get("feedback_index", 0)
                            category_indices = classification.get("category_indices", [])
                            
                            if feedback_idx < len(chunk_texts):
                                feedback_text = chunk_texts[feedback_idx]
                                
                                # If no categories matched, assign to first category
                                if not category_indices and key_areas:
                                    first_area = key_areas[0].get('area', 'Area 1')
                                    classified_feedback[first_area].append(feedback_text)
                                else:
                                    # Add to all matching categories
                                    for cat_idx in category_indices:
                                        if 0 <= cat_idx < len(key_areas):
                                            area = key_areas[cat_idx].get('area', f'Area {cat_idx+1}')
                                            classified_feedback[area].append(feedback_text)
                
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i//chunk_size + 1}: {str(chunk_error)}")
                    # Assign all feedbacks in this chunk to the first category as fallback
                    if key_areas:
                        first_area = key_areas[0].get('area', 'Area 1')
                        classified_feedback[first_area].extend(chunk_texts)
                
                # Avoid rate limits
                await asyncio.sleep(0.5)
        
        # Log classification stats
        logger.info("Classification results:")
        for area, feedbacks in classified_feedback.items():
            logger.info(f"  - {area}: {len(feedbacks)} items")
        
        # Create a mapping from text to item and add sentiment scores
        text_to_item = {item["text"]: item for item in all_feedback_items}
        enriched_feedback = {}
        
        for area, feedbacks in classified_feedback.items():
            enriched_feedback[area] = []
            for feedback_text in feedbacks:
                if feedback_text in text_to_item:
                    # Include original row data
                    original_item = text_to_item[feedback_text]
                    original_row = original_item.get("original_row", {"text": feedback_text})
                    
                    # Add sentiment score if available
                    if feedback_text in text_to_sentiment:
                        original_row["sentiment_score"] = text_to_sentiment[feedback_text]
                    else:
                        # Use a neutral sentiment as fallback
                        original_row["sentiment_score"] = 0.0
                        
                    enriched_feedback[area].append(original_row)
                else:
                    # Fallback if text not found in mapping
                    enriched_feedback[area].append({"text": feedback_text, "sentiment_score": 0.0})
        
        # Replace classified_feedback with enriched version that includes sentiment
        classified_feedback = enriched_feedback

        return {
            "key_areas": key_areas,
            "classified_feedback": classified_feedback
        }
        
    except Exception as e:
        logger.error(f"Error processing Excel data: {str(e)}\n{traceback.format_exc()}")
        # Fall back to raw data processing if Excel processing fails
        try:
            # Convert Excel to CSV format as fallback
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
                temp.write(file_content)
                temp_path = temp.name
            
            # Read the first sheet as fallback
            df = pd.read_excel(temp_path, sheet_name=0)
            csv_data = df.to_csv(index=False)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            logger.info("Falling back to CSV processing for Excel data")
            return await process_csv_data(AZURE_CLIENT, csv_data, source)
        except Exception as fallback_error:
            logger.error(f"Excel fallback also failed: {str(fallback_error)}")
            # Final fallback - raw data
            logger.info("Falling back to raw data processing")
            excel_text = str(file_content)  # Very crude fallback
            return await analyze_raw_data_chunks(AZURE_CLIENT, excel_text)
async def identify_key_areas(client: AzureOpenAI, data_sample: str, max_areas: int = 15) -> List[Dict[str, str]]:
    """
    Identifies key problem areas from customer feedback using OpenAI.
    
    Args:
        client: Azure OpenAI client
        data_sample: Sample of feedback data to analyze
        max_areas: Maximum number of key areas to identify
        
    Returns:
        List of dictionaries with 'area' and 'problem' keys
    """
    try:
        logger.info(f"Beginning key problem area identification using {len(data_sample)} characters of sample data")
    
        # Prepare the prompt for OpenAI
        prompt = f"""
        # Customer Feedback Analysis Task

        You are a senior product insights analyst specializing in customer feedback pattern recognition. 
        Your expertise lies in identifying underlying themes and problems from diverse customer feedback.
        You are also expert in extracting app/ui specific problems which ruin the customer experience.

        ## Your Objective
        Analyze the provided customer feedback data and identify the most significant problem areas that customers are experiencing, with a focus on user experience, app-related issues while still capturing important physical/in-store issues when relevant. These insights will directly inform product development priorities.

        ## Customer Feedback Data Sample
        ```
        {data_sample}
        ```
        ## Analysis Instructions:
        1. Carefully read and understand all customer feedback in the sample

        2. Identify the top {max_areas} most significant problem areas based on:
        - Frequency (how many customers mention this issue)
        - Severity (how impactful the problem seems to be)
        - Specificity (clear, actionable problem statements)
        - Business impact (issues affecting core product value)

        3. For each identified problem area:
        - Create a concise, high-level area (2-3 words) reflecting the functional area
        - Write a specific problem statement from the customer's perspective (1 sentence)
        - Ensure the problem is concrete enough to be actionable
        - Capture the essence of what customers are struggling with
        - Note that the same high-level area (e.g., "App Performance") can have multiple specific problems

        4. You may:
        - Select directly from the reference categories if they match well
        - Adapt a reference category with more specific wording
        - Create entirely new categories if the existing ones don't capture the feedback
        - Combine similar issues into a single coherent problem area
        - Group related issues under the same area with different problem statements

        ## Response Format Requirements
        You must format your response as a JSON array with each problem area having 'area' and 'problem' keys:

        ```json
        [
            {{"area": "Performance", "problem": "App frequently crashes when switching between multiple workout screens"}},
            {{"area": "Performance", "problem": "Physical device overheats during extended outdoor training sessions"}},
            {{"area": "User Interface", "problem": "App navigation requires too many taps to access core tracking features"}},,
            {{"area": "User Interface", "problem": "Physical buttons on the device are difficult to press while wearing gloves"}},
            {{"area": "Data Management", "problem": "App loses workout history when syncing with cloud services"}},
            {{"area": "Data Management", "problem": "Limited onboard storage capacity prevents saving longer activity sessions"}},
            {{"area": "Connectivity", "problem": "App fails to maintain Bluetooth connection with heart rate monitors"}},
            {{"area": "Connectivity", "problem": "Device requires proprietary cables that are difficult to replace when damaged"}},
            {{"area": "Tracking Accuracy", "problem": "App calorie calculations show inconsistent results compared to similar services"}},
            {{"area": "Tracking Accuracy", "problem": "Physical sensors produce erratic readings during high-intensity workouts"}},
            {{"area": "Customization", "problem": "App provides limited options for personalizing workout routines"}},
            {{"area": "Customization", "problem": "Device band sizing options don't accommodate larger or smaller wrists"}},
            {{"area": "Battery Life", "problem": "App continues to drain battery significantly when running in background"}},
            {{"area": "Battery Life", "problem": "Physical device requires charging after every workout session"}},
            {{"area": "Customer Support", "problem": "App troubleshooting guides don't address common synchronization problems"}},
            {{"area": "Customer Support", "problem": "Replacement parts for physical device have extensive shipping delays"}}
        ]
        ```

        Think carefully about naming each area - these need to be specific enough to be meaningful but general enough to group similar issues.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a product analytics specialist who identifies key problem areas from customer feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        result_text = response.choices[0].message.content
        try:
            result = json.loads(result_text)
            
            # Handle different possible response formats
            if isinstance(result, dict):
                # Check for common response structures
                if "response" in result:
                    key_areas = result["response"]
                    logger.info("Found key areas in 'response' field")
                elif "areas" in result:
                    key_areas = result["areas"]
                    logger.info("Found key areas in 'areas' field")
                elif "key_areas" in result:
                    key_areas = result["key_areas"]
                    logger.info("Found key areas in 'key_areas' field")
                # If no recognized fields but has area/problem structure, use the whole object
                elif result.get("area") and result.get("problem"):
                    key_areas = [result]
                    logger.info("Found single key area object")
                else:
                    # Try to extract any list with area/problem structure
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            if "area" in value[0] and "problem" in value[0]:
                                key_areas = value
                                logger.info(f"Found key areas in '{key}' field")
                                break
                    else:
                        logger.warning(f"Unrecognized dictionary structure in OpenAI response, using entire object")
                        key_areas = [result]
            elif isinstance(result, list):
                # Check if list items have area/problem structure
                if result and isinstance(result[0], dict) and "area" in result[0] and "problem" in result[0]:
                    key_areas = result
                    logger.info("Found key areas in list format")
                else:
                    logger.warning("List items in response don't have expected structure")
                    key_areas = []
            else:
                logger.warning(f"Response is neither dict nor list: {type(result)}")
                key_areas = []
                
            if not key_areas:
                logger.warning(f"Could not extract key areas from response: {result_text[:500]}...")
                key_areas = []
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {result_text[:500]}...")
            key_areas = []
            
        # Log the raw response if we had issues
        if not key_areas:
            logger.warning(f"Raw response from OpenAI: {result_text[:1000]}...")
            # Fallback to common key areas
            key_areas = COMMON_KEY_AREAS[:max_areas]
            logger.info(f"Using {len(key_areas)} fallback key areas")
            
        # Limit to max_areas and ensure consistent structure
        key_areas = key_areas[:max_areas]
        
        # Ensure each area has both 'area' and 'problem' fields
        for area in key_areas:
            if not isinstance(area, dict):
                logger.warning(f"Key area is not a dictionary: {area}")
                continue
            if "area" not in area:
                area["area"] = "Unknown Area"
            if "problem" not in area:
                area["problem"] = "Unspecified problem"
        
        # Log the identified key areas in detail
        logger.info(f"IDENTIFIED {len(key_areas)} KEY PROBLEM AREAS:")
        for i, area in enumerate(key_areas):
            logger.info(f"  {i+1}. {area['area']}: {area['problem']}")
        return key_areas
        
    except Exception as e:
        logging.error(f"Error identifying key areas: {str(e)}")
        # Return a minimal structure to continue processing
        return [{"area": "General Issues", "problem": "Various customer problems and feedback"}]

async def get_embeddings(client: AzureOpenAI, texts: List[str], slow_mode: bool = False) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        client: Azure OpenAI client
        texts: List of text strings to generate embeddings for
        slow_mode: If True, use the original sequential processing
        
    Returns:
        List of embedding vectors
    """
    try:
        # Process in batches to avoid request size limits
        batch_size = 100 if slow_mode else 1000
        all_embeddings = []
        
        logger.info(f"Generating embeddings for {len(texts)} texts using model {EMBEDDING_MODEL}")
        
        if slow_mode:
            # Original sequential implementation
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    response = AZURE_CLIENT.embeddings.create(
                        model=EMBEDDING_MODEL,  # Use specific model name
                        input=batch_texts
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    logger.info(f"Successfully generated {len(batch_embeddings)} embeddings (batch {i//batch_size + 1})")
                    
                    # Check embedding dimension for debugging
                    if batch_embeddings:
                        dim = len(batch_embeddings[0])
                        logger.info(f"Embedding dimension: {dim}")
                    
                    # Avoid rate limits
                    if i + batch_size < len(texts):
                        await asyncio.sleep(0.5)
                        
                except Exception as batch_error:
                    logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(batch_error)}")
                    # For this batch, generate random embeddings as fallback
                    # Use fixed dimension of 3072 to match the model
                    dim = 1536     # text-embedding-3-large dimension
                    logger.warning(f"Using fallback random embeddings for batch {i//batch_size + 1} with dimension {dim}")
                    for _ in range(len(batch_texts)):
                        random_embedding = list(np.random.normal(0, 0.1, dim))
                        all_embeddings.append(random_embedding)
        else:
            # Optimized parallel implementation
            async def process_batch(batch_idx, batch_texts):
                try:
                    response = AZURE_CLIENT.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    logger.info(f"Successfully generated {len(batch_embeddings)} embeddings (batch {batch_idx+1})")
                    return batch_embeddings
                except Exception as batch_error:
                    logger.error(f"Error generating embeddings for batch {batch_idx+1}: {str(batch_error)}")
                    dim = 1536   # text-embedding-3-large dimension
                    return [list(np.random.normal(0, 0.1, dim)) for _ in range(len(batch_texts))]
            
            # Split texts into batches
            batches = [(i//batch_size, texts[i:i + batch_size]) 
                       for i in range(0, len(texts), batch_size)]
            
            # Process batches in parallel with rate limiting
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls
            
            async def process_with_semaphore(batch_idx, batch_texts):
                async with semaphore:
                    return await process_batch(batch_idx, batch_texts)
            
            # Gather results from all batches
            batch_results = await asyncio.gather(
                *[process_with_semaphore(idx, batch) for idx, batch in batches]
            )
            
            # Flatten results
            for batch_emb in batch_results:
                all_embeddings.extend(batch_emb)
        
        return all_embeddings
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Generate random embeddings as fallback
        logger.warning("Using fallback random embeddings for all texts")
        dim = 1536   # text-embedding-3-large dimension
        return [list(np.random.normal(0, 0.1, dim)) for _ in range(len(texts))]
def cosine_similarity_batch(A: np.ndarray, B: np.ndarray, slow_mode: bool = False) -> np.ndarray:
    """
    Calculate cosine similarity between two sets of vectors efficiently.
    
    Args:
        A: Matrix of shape (n_samples_A, n_features)
        B: Matrix of shape (n_samples_B, n_features)
        slow_mode: If True, use a slower for-loop implementation
        
    Returns:
        Similarity matrix of shape (n_samples_A, n_samples_B)
    """
    if slow_mode:
        # Slower iterative implementation (for compatibility)
        n_samples_A, _ = A.shape
        n_samples_B, _ = B.shape
        similarity_matrix = np.zeros((n_samples_A, n_samples_B))
        
        for i in range(n_samples_A):
            for j in range(n_samples_B):
                dot_product = np.dot(A[i], B[j])
                norm_A = np.linalg.norm(A[i])
                norm_B = np.linalg.norm(B[j])
                
                if norm_A == 0 or norm_B == 0:
                    similarity_matrix[i, j] = 0
                else:
                    similarity_matrix[i, j] = dot_product / (norm_A * norm_B)
        
        return similarity_matrix
    else:
        # Fast vectorized implementation
        # Normalize the matrices
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        # Calculate similarity matrix in one operation
        return np.dot(A_norm, B_norm.T)
async def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    # Convert to numpy arrays for vectorized operations
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Calculate dot product and norms
    dot_product = np.dot(vec1_np, vec2_np)
    norm_a = np.linalg.norm(vec1_np)
    norm_b = np.linalg.norm(vec2_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

async def identify_relevant_columns(client: AzureOpenAI, df: pd.DataFrame) -> Dict[str, str]:
    """
    Uses OpenAI to identify which columns contain ratings, feedback text, received date, name, and source.
    
    Args:
        client: Azure OpenAI client
        df: Pandas DataFrame containing the data
        
    Returns:
        Dictionary with column mappings
    """
    try:
        logger.info(f"Beginning column detection on DataFrame with shape: {df.shape}")
        logger.info(f"Available columns: {list(df.columns)}")
        
        # Get a sample of the data
        sample = df.sample(min(5, len(df))).to_csv(index=False)
        columns_list = list(df.columns)
        columns_with_indices = [f"{i}: {col}" for i, col in enumerate(columns_list)]
        columns_text = "\n".join(columns_with_indices)
        
        prompt = f"""
        You are a specialized data analyst focusing on customer feedback analysis. Your task is to precisely identify the most relevant columns in this customer feedback dataset.

        DATASET SAMPLE:
        ```
        {sample}
        ```

        AVAILABLE COLUMNS (with indices):
        ```
        {columns_text}
        ```

        TASK:
        Carefully examine the dataset and identify these types of columns:

        1. CUSTOMER FEEDBACK COLUMN:
           - Contains textual customer opinions, comments, reviews, or feedback
           - Usually includes sentences, paragraphs, or detailed comments
           - May contain terms like "liked", "disliked", "issue with", "problem", etc.
           - Look for the column with the most detailed textual content
        
        2. RATING COLUMN:
           - Contains numerical scores (1-5, 1-10) or textual ratings ("Excellent", "Poor")
           - Often named "rating", "score", "stars", "satisfaction", etc.
           - May be presented as numbers or categorical values

        3. RECEIVED COLUMN:
           - Contains date or timestamp information when the feedback was received
           - Often named "date", "received", "submitted", "timestamp", etc.
           - May be formatted as date, datetime, or timestamp

        4. NAME COLUMN:
           - Contains the name of the reviewer or customer
           - Often named "name", "customer", "reviewer", "user", etc.
           - Usually contains full names, first names, or usernames

        5. SOURCE COLUMN:
           - Contains information about where the feedback came from
           - Often named "source", "platform", "channel", "website", etc.
           - May contain values like "Google", "Amazon", "Website", "App", etc.

        For each column you identify, consider:
        - Column content and data type
        - Column name relevance
        - Amount of information in the column
        - Uniqueness of values

        IMPORTANT INSTRUCTIONS:
        - You MUST select from the provided column indices (0 to {len(columns_list)-1})
        - You must specify both the index and exact column name
        - If a certain type of column doesn't exist, set the value to null

        RESPONSE FORMAT:
        You must respond with column indices and names in this exact JSON format:
        {{
            "feedback_col": "3: [Exact Column Name]",
            "rating_col": "1: [Exact Column Name]",
            "received_col": "0: [Exact Column Name]",
            "name_col": "2: [Exact Column Name]",
            "source_col": "4: [Exact Column Name]"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant that identifies column types in customer feedback datasets."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Extract column information and handle fuzzy matching if needed
        columns = {
            "feedback_col": None, 
            "rating_col": None,
            "received_col": None,
            "name_col": None,
            "source_col": None
        }
        
        for col_type in columns.keys():
            if result.get(col_type):
                # Parse the response format "index: Column Name"
                try:
                    col_parts = result[col_type].split(":", 1)
                    if len(col_parts) == 2:
                        idx = int(col_parts[0].strip())
                        col_name = col_parts[1].strip()
                        
                        # Verify the column exists
                        if 0 <= idx < len(columns_list) and columns_list[idx] == col_name:
                            columns[col_type] = col_name
                        else:
                            # Try fuzzy matching
                            best_match, score = process.extractOne(col_name, columns_list)
                            if score > 80:  # Good match threshold
                                columns[col_type] = best_match
                            else:
                                logging.warning(f"Column '{col_name}' not found and no good match")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Error parsing column specification '{result[col_type]}': {e}")
        
        # Apply fallback strategies for each column type
        # Feedback column fallback
        if not columns["feedback_col"]:
            # Try to find a text column with keywords
            keywords = ["comment", "feedback", "review", "text", "description", "comments"]
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    columns["feedback_col"] = col
                    break
            
            # If still not found, use first string column with sufficient data
            if not columns["feedback_col"]:
                logger.warning("No feedback column identified via AI or keyword matching - using heuristics")
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].notna().sum() > len(df) * 0.5:
                        logger.info(f"Selected '{col}' as feedback column based on data type and non-null values")
                        columns["feedback_col"] = col
                        # Log sample values for verification
                        sample_values = df[col].dropna().sample(min(3, df[col].notna().sum())).tolist()
                        logger.info(f"Sample feedback values: {sample_values}")
                        break
        
        # Rating column fallback
        if not columns["rating_col"]:
            # Try to find a numeric column with keywords
            keywords = ["rating", "score", "stars", "grade", "rank"]
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    logger.info(f"Selected '{col}' as rating column based on keyword match")
                    columns["rating_col"] = col
                    break
            
            # If still not found, use first numeric column
            if not columns["rating_col"]:
                logger.warning("No rating column identified via keywords - looking for numeric columns")
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        sample_values = df[col].dropna().sample(min(5, df[col].notna().sum())).tolist()
                        # Check if values look like ratings (typically 1-5, 1-10)
                        if all(isinstance(val, (int, float)) for val in sample_values) and all(0 <= val <= 10 for val in sample_values):
                            logger.info(f"Selected '{col}' as rating column based on numeric values in typical rating range")
                            columns["rating_col"] = col
                            break
                            
        # Received column fallback
        if not columns["received_col"]:
            # Try to find date/time columns
            keywords = ["date", "time", "received", "created", "timestamp", "submitted"]
            
            # First check column names
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    logger.info(f"Selected '{col}' as received column based on keyword match")
                    columns["received_col"] = col
                    break
            
            # If still not found, check for datetime-like columns
            if not columns["received_col"]:
                for col in df.columns:
                    # Check if column has date-like values
                    if df[col].dtype == 'datetime64[ns]':
                        logger.info(f"Selected '{col}' as received column based on datetime type")
                        columns["received_col"] = col
                        break
                    elif df[col].dtype == 'object':
                        # Check if strings might be dates
                        sample = df[col].dropna().sample(min(5, df[col].notna().sum())).tolist()
                        date_patterns = [
                            r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # yyyy-mm-dd, mm/dd/yyyy
                            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # dd/mm/yyyy, mm-dd-yy
                            r'\w+\s+\d{1,2},?\s+\d{2,4}'      # Month Day, Year
                        ]
                        if any(sample) and any(re.search(pattern, str(s)) for s in sample for pattern in date_patterns):
                            logger.info(f"Selected '{col}' as received column based on date-like strings")
                            columns["received_col"] = col
                            break
                            
        # Name column fallback
        if not columns["name_col"]:
            keywords = ["name", "customer", "reviewer", "user", "client", "person"]
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    logger.info(f"Selected '{col}' as name column based on keyword match")
                    columns["name_col"] = col
                    break
                    
        # Source column fallback
        if not columns["source_col"]:
            keywords = ["source", "platform", "channel", "origin", "website", "site"]
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in keywords):
                    logger.info(f"Selected '{col}' as source column based on keyword match")
                    columns["source_col"] = col
                    break
        
        logger.info(f"COLUMN DETECTION COMPLETE: feedback={columns['feedback_col']}, rating={columns['rating_col']}, received={columns['received_col']}, name={columns['name_col']}, source={columns['source_col']}")
        
        return columns
        
    except Exception as e:
        logging.error(f"Error identifying columns: {str(e)}")
        # Return empty result, will fall back to heuristics
        return {
            "feedback_col": None, 
            "rating_col": None,
            "received_col": None,
            "name_col": None,
            "source_col": None
        }

async def analyze_raw_data_chunks(
    client: AzureOpenAI, 
    raw_data: str,
    chunk_size: int = 8000,
    overlap: int = 500,
    slow_mode: bool = False
) -> Dict[str, Any]:
    """
    Analyzes raw unstructured data by chunking and processing with OpenAI.
    
    Args:
        client: Azure OpenAI client
        raw_data: Raw text data
        chunk_size: Size of chunks to process
        overlap: Overlap between chunks
        slow_mode: If True, use the original sequential processing
        
    Returns:
        Dictionary with key areas and classified feedback
    """
    try:
        # Split data into chunks
        chunks = []
        for i in range(0, len(raw_data), chunk_size - overlap):
            chunk = raw_data[i:i + chunk_size]
            chunks.append(chunk)
        
        logger.info(f"Split raw data into {len(chunks)} chunks for processing")
        
        # STAGE 1: Extract feedback items and preliminary areas from chunks
        all_feedbacks = []
        chunk_areas = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Extract feedback from chunk
            prompt = f"""
            Analyze this raw customer feedback data:
            
            {chunk}
            
            1. Identify distinct customer feedback items/comments.
            2. For each feedback item, extract the exact text.
            3. Ignore any 5-star or extremely positive reviews.
            
            Format as a JSON array of feedback items.
            Example: ["I can't generate the reports I need", "The mobile app lacks key features"]
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are a customer feedback analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                # Extract feedback items
                if isinstance(result, dict) and "feedback" in result:
                    feedbacks = result["feedback"]
                elif isinstance(result, list):
                    feedbacks = result
                else:
                    feedbacks = []
                
                # Store feedbacks with original index and chunk info
                for j, feedback in enumerate(feedbacks):
                    all_feedbacks.append({
                        "text": feedback,
                        "original_index": len(all_feedbacks),
                        "chunk_index": i
                    })
                
                # Identify preliminary areas in this chunk
                if feedbacks:
                    sample_feedback = '\n'.join(feedbacks[:50])
                    areas = await identify_key_areas(AZURE_CLIENT, sample_feedback, max_areas=15)
                    chunk_areas.extend(areas)
                
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i+1}: {str(chunk_error)}")
                continue
            
            # Avoid rate limits
            await asyncio.sleep(1)
        
        # STAGE 2: Consolidate problem areas across all chunks
        if chunk_areas:
            # Use OpenAI to consolidate
            areas_text = json.dumps(chunk_areas, indent=2)
            prompt = f"""
            I have identified multiple potential problem areas from customer feedback chunks. 
            Please consolidate these into 5-8 main categories.
            
            {areas_text}
            
            For each category:
            1. Provide a short, consistent name (2-3 words)
            2. Provide a specific problem statement from the customer's perspective
            
            Format your response as a JSON array of objects with 'area' and 'problem' keys:
            [
                {{"area": "User Interface", "problem": "The app navigation requires too many taps to access core features"}},
                {{"area": "Performance", "problem": "The app frequently crashes during extended usage sessions"}}
            ]
            """
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a product analytics specialist who identifies key problem areas from customer feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Extract final areas
            final_areas = []
            if isinstance(result, dict) and any(isinstance(v, list) for v in result.values()):
                for k, v in result.items():
                    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                        if all('area' in x for x in v):
                            final_areas = v
                            break
            elif isinstance(result, list) and all(isinstance(x, dict) for x in result):
                if all('area' in x for x in result):
                    final_areas = result
            
            # If we couldn't parse the consolidated areas properly, use the raw chunk areas
            if not final_areas:
                logger.warning("Could not parse consolidated areas, using raw chunk areas")
                # Try to handle both dictionary and string formats
                if chunk_areas and isinstance(chunk_areas[0], dict) and 'area' in chunk_areas[0]:
                    final_areas = chunk_areas
                else:
                    # Convert strings to dictionaries
                    final_areas = [{"area": area, "problem": f"Issues related to {area.lower()}"} 
                                  for area in chunk_areas if isinstance(area, str)]
        else:
            # Create default areas if none were identified
            final_areas = [
                {"area": "General Issues", "problem": "Various general problems with the service or product"},
                {"area": "Equipment Problems", "problem": "Issues with physical equipment or hardware"},
                {"area": "Staff & Service", "problem": "Problems with staff behavior or service quality"},
                {"area": "Facility Maintenance", "problem": "Issues with facility cleanliness or maintenance"}
            ]
        
        logger.info(f"Consolidated into {len(final_areas)} final areas")
        for i, area in enumerate(final_areas):
            logger.info(f"  {i+1}. {area.get('area')}: {area.get('problem')}")
            
        # STAGE 3: Use embeddings for fast classification
        # Generate embeddings for key areas - use only the area names for embedding
        problem_texts = [area.get('problem', f"Problem related to {area.get('area', 'unknown')}") for area in final_areas]
        key_area_embeddings = await get_embeddings(AZURE_CLIENT, problem_texts)
        area_names = [area.get('area') for area in final_areas]
        
        # Generate embeddings for all feedback
        feedback_texts = [item["text"] for item in all_feedbacks]
        
        # Process in batches to handle potential size limitations
        batch_size = 300
        all_feedback_embeddings = []
        
        for i in range(0, len(feedback_texts), batch_size):
            batch = feedback_texts[i:i + batch_size]
            logger.info(f"Generating embeddings for feedback batch {i//batch_size + 1}/{(len(feedback_texts) + batch_size - 1)//batch_size}")
            batch_embeddings = await get_embeddings(AZURE_CLIENT, batch)
            all_feedback_embeddings.extend(batch_embeddings)
        
        # Calculate sentiment scores using the same embeddings we already generated
        logger.info("Calculating sentiment scores for all feedback items")
        sentiment_scores = await calculate_sentiment_scores(all_feedback_embeddings, client)
        
        # Create a mapping from text to sentiment score
        text_to_sentiment = {feedback_texts[i]: sentiment_scores[i] for i in range(len(feedback_texts))}
        
        # Function to calculate cosine similarity matrix
        def cosine_similarity_matrix(A, B):
            # Normalize the matrices
            A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
            B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
            # Calculate similarity matrix
            similarity = np.dot(A_norm, B_norm.T)
            return similarity
        
        # Calculate similarity between each feedback and each key area
        logger.info("Calculating similarity matrix")
        key_area_matrix = np.array(key_area_embeddings)
        feedback_matrix = np.array(all_feedback_embeddings)
        
        # Verify dimensions match
        key_area_dim = key_area_matrix.shape[1]
        feedback_dim = feedback_matrix.shape[1]
        logger.info(f"Embedding dimensions - Key areas: {key_area_matrix.shape}, Feedback: {feedback_matrix.shape}")
        
        if key_area_dim != feedback_dim:
            logger.error(f"Embedding dimensions mismatch: key_areas({key_area_dim}) != feedback({feedback_dim})")
            raise ValueError(f"Embedding dimensions don't match: {key_area_dim} vs {feedback_dim}")
            
        similarity_matrix = cosine_similarity_matrix(feedback_matrix, key_area_matrix)
        
        # Classify feedback based on similarity
        similarity_threshold = 0.3  # Lower threshold to match more feedback
        classified_feedback = {area.get('area'): [] for area in final_areas}
        
        for i, similarities in enumerate(similarity_matrix):
            # Get indices where similarity exceeds threshold
            matches = np.where(similarities > similarity_threshold)[0]
            
            # If no matches, use the best match
            if len(matches) == 0:
                best_match = np.argmax(similarities)
                matches = [best_match]
            
            # Add feedback to all matching areas
            for match_idx in matches:
                if match_idx < len(final_areas):
                    area = final_areas[match_idx].get('area')
                    classified_feedback[area].append(feedback_texts[i])
        
        logger.info(f"Classified {len(all_feedbacks)} feedback items into {len(final_areas)} areas")
        text_to_item = {item["text"]: item for item in all_feedbacks}
        def clean_json_values(obj):
            if isinstance(obj, dict):
                return {k: clean_json_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_json_values(i) for i in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return -1
            else:
                return obj
                
        enriched_feedback = {}
        
        for area, feedbacks in classified_feedback.items():
            enriched_feedback[area] = []
            for feedback_text in feedbacks:
                if feedback_text in text_to_item:
                    # Include original row data
                    original_item = text_to_item[feedback_text]
                    original_row = original_item.get("original_row", {"text": feedback_text})
                    original_row = clean_json_values(original_row)
                    
                    # Add sentiment score if available
                    if feedback_text in text_to_sentiment:
                        original_row["sentiment_score"] = text_to_sentiment[feedback_text]
                    else:
                        # Use a neutral sentiment as fallback
                        original_row["sentiment_score"] = 0.0
                        
                    enriched_feedback[area].append(original_row)
                else:
                    # Fallback if text not found in mapping
                    enriched_feedback[area].append({"text": feedback_text, "sentiment_score": 0.0})
        
        # Replace classified_feedback with enriched version
        classified_feedback = enriched_feedback
        return {
            "key_areas": final_areas,  # Return the full dictionary structure
            "classified_feedback": classified_feedback
        }
        
    except Exception as e:
        logger.error(f"Error in raw data analysis: {str(e)}\n{traceback.format_exc()}")
        # Create default areas if exception occurs
        basic_areas = [
            {"area": "General Issues", "problem": "Various general problems with the service or product"},
            {"area": "Equipment Problems", "problem": "Issues with physical equipment or hardware"},
            {"area": "Staff & Service", "problem": "Problems with staff behavior or service quality"},
            {"area": "Facility Maintenance", "problem": "Issues with facility cleanliness or maintenance"}
        ]
        # Create a simple fallback classification
        feedback_by_area = {}
        for i, area in enumerate(basic_areas):
            area_name = area.get('area')
            start_idx = (len(all_feedbacks) * i) // len(basic_areas)
            end_idx = (len(all_feedbacks) * (i+1)) // len(basic_areas)
            feedback_by_area[area_name] = [{"text": item["text"], "sentiment_score": 0.0} for item in all_feedbacks[start_idx:end_idx]]
        
        return {
            "key_areas": basic_areas,
            "classified_feedback": feedback_by_area
        }

async def process_csv_data(
    client: AzureOpenAI, 
    csv_data: str,
    source: str = None,  # Add optional source parameter
    slow_mode: bool = False
) -> Dict[str, Any]:
    """
    Process structured CSV data to extract and classify feedback.
    
    Args:
        client: Azure OpenAI client
        csv_data: CSV data as string
        source: Optional source value to override the Source column
        slow_mode: If True, use the original sequential processing
        
    Returns:
        Dictionary with key areas and classified feedback
    """
    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))
        
        if df.empty or len(df.columns) == 0:
            raise ValueError("Empty or invalid CSV data")
            
        logger.info(f"Parsed CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Use OpenAI to identify relevant columns
        columns = await identify_relevant_columns(AZURE_CLIENT, df)
        feedback_col = columns.get("feedback_col")
        rating_col = columns.get("rating_col")
        
        logger.info(f"Using columns: feedback={feedback_col}, rating={rating_col}")
        
        # Standardize the dataframe with the optional source parameter
        if feedback_col and feedback_col in df.columns:
            df = standardize_dataframe(df, columns, source)
            # Update column references after standardization
            feedback_col = "Customer Feedback"
            # If rating column was renamed, update the reference
            if rating_col in columns and columns[rating_col] == feedback_col:
                rating_col = None
        
        # Extract feedback items, handling NaN values
        if feedback_col and feedback_col in df.columns:
            # Skip 5-star reviews if rating column exists
            if rating_col and rating_col in df.columns:
                try:
                    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
                    high_rating_mask = df[rating_col] >= 5
                    five_star_count = high_rating_mask.sum()
                    logger.info(f"Skipping {five_star_count} high-rated reviews (5-star)")
                    df = df[~high_rating_mask]
                except:
                    logger.warning("Could not filter out 5-star reviews, processing all")
            
            # Extract feedback with ratings - vectorized approach
            valid_rows = df[df[feedback_col].notna() & (df[feedback_col].astype(str).str.strip() != "")]
            all_feedback_items = []
            for idx, row in valid_rows.iterrows():
                feedback_text = str(row[feedback_col])
                if rating_col and rating_col in df.columns and pd.notna(row[rating_col]):
                    feedback_text = f"[Rating: {row[rating_col]}] {feedback_text}"
                row_dict = row.replace([float('nan'), float('inf'), float('-inf')], -1).to_dict()
                all_feedback_items.append({
                    "text": feedback_text,
                    "original_index": idx,
                    "original_row": row_dict
                })
        else:
            # If no feedback column identified, use the first text column
            all_feedback_items = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    valid_rows = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
                    for idx, row in valid_rows.iterrows():
                        all_feedback_items.append({
                            "text": str(row[col]),
                            "original_index": idx,
                            "original_row": row.to_dict()
                        })
                    break
        
        if not all_feedback_items:
            raise ValueError("No valid feedback found in the CSV data")
            
        logger.info(f"Extracted {len(all_feedback_items)} feedback items")
            
        # Sample for key area identification
        sample_feedbacks = [item["text"] for item in all_feedback_items[:min(1000, len(all_feedback_items) // 2)]]
        sample_feedback_text = '\n'.join(sample_feedbacks)
        
        # Identify key areas in parallel with other initialization
        key_area_task = asyncio.create_task(identify_key_areas(client, sample_feedback_text))
        
        # Process feedback in batches for embedding - start while key areas are being identified
        feedback_texts = [item["text"] for item in all_feedback_items]
        
        # Get key areas from the parallel task
        key_areas = await key_area_task
        logger.info(f"Identified {len(key_areas)} key areas")
        
        # Initialize result structure with empty lists - use the area name as key
        classified_feedback = {area.get('area', f"Area {i+1}"): [] for i, area in enumerate(key_areas)}
        
        # Use embeddings for classification
        try:
            logger.info("Using embeddings for feedback classification")
            
            # Create a list of just the area names for embeddings
            problem_texts = [area.get('problem', f"Problem {i+1}") for i, area in enumerate(key_areas)]
            key_area_embeddings_task = asyncio.create_task(get_embeddings(client, problem_texts))
            area_names = [area.get('area', f"Area {i+1}") for i, area in enumerate(key_areas)]
            
            # Process feedback in batches for embedding - parallelize with area embeddings
            batch_size = 300
            feedback_embeddings_tasks = []
            
            for i in range(0, len(feedback_texts), batch_size):
                batch = feedback_texts[i:i + batch_size]
                logger.info(f"Generating embeddings for feedback batch {i//batch_size + 1}/{(len(feedback_texts) + batch_size - 1)//batch_size}")
                task = asyncio.create_task(get_embeddings(client, batch))
                feedback_embeddings_tasks.append(task)
            
            # Wait for all embedding tasks to complete
            all_embedding_tasks = [key_area_embeddings_task] + feedback_embeddings_tasks
            embedding_results = await asyncio.gather(*all_embedding_tasks)
            
            # First result is key area embeddings, rest are feedback embedding batches
            key_area_embeddings = embedding_results[0]
            all_feedback_embeddings = []
            for result in embedding_results[1:]:
                all_feedback_embeddings.extend(result)
            
            # Calculate sentiment scores using the same embeddings we already generated
            logger.info("Calculating sentiment scores for all feedback items")
            sentiment_scores = await calculate_sentiment_scores(all_feedback_embeddings, client)
            
            # Create a mapping from text to sentiment score
            text_to_sentiment = {feedback_texts[i]: sentiment_scores[i] for i in range(len(feedback_texts))}
            
            # Calculate similarity and classify using numpy - vectorized operations
            key_area_matrix = np.array(key_area_embeddings)
            feedback_matrix = np.array(all_feedback_embeddings)
            
            # Verify dimensions match
            key_area_dim = key_area_matrix.shape[1]
            feedback_dim = feedback_matrix.shape[1]
            logger.info(f"Embedding dimensions - Key areas: {key_area_matrix.shape}, Feedback: {feedback_matrix.shape}")
            
            if key_area_dim != feedback_dim:
                logger.error(f"Embedding dimensions mismatch: key_areas({key_area_dim}) != feedback({feedback_dim})")
                raise ValueError(f"Embedding dimensions don't match: {key_area_dim} vs {feedback_dim}")
            
            # Calculate similarity with optimized matrix operations
            A_norm = feedback_matrix / np.linalg.norm(feedback_matrix, axis=1, keepdims=True)
            B_norm = key_area_matrix / np.linalg.norm(key_area_matrix, axis=1, keepdims=True)
            similarity_matrix = np.dot(A_norm, B_norm.T)
            
            # Classify feedback based on similarity
            similarity_threshold = 0.7
            
            # Vectorized matching - much faster than looping
            matches = similarity_matrix > similarity_threshold
            
            # For rows with no matches, use the best match
            no_matches = ~np.any(matches, axis=1)
            if np.any(no_matches):
                best_matches = np.argmax(similarity_matrix[no_matches], axis=1)
                for i, row_idx in enumerate(np.where(no_matches)[0]):
                    matches[row_idx, best_matches[i]] = True
            
            # Add feedback to matching areas using the matches matrix
            for i in range(matches.shape[0]):
                for j in np.where(matches[i])[0]:
                    if j < len(area_names):
                        area = area_names[j]
                        classified_feedback[area].append(feedback_texts[i])
            
            logger.info(f"Successfully classified {len(all_feedback_embeddings)} feedback items using embeddings")
            
        except Exception as e:
            logger.error(f"Error with embedding classification: {str(e)}")
            logger.info("Falling back to direct OpenAI classification")
            
            # Fall back to chunked classification via OpenAI
            chunk_size = 10
            
            # Using semaphore to limit concurrent API calls
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent API calls
            
            async def process_chunk(chunk_idx, chunk):
                async with semaphore:
                    logger.info(f"Processing feedback chunk {chunk_idx + 1}/{(len(all_feedback_items) + chunk_size - 1)//chunk_size}")
                    
                    chunk_texts = [item["text"] for item in chunk]
                    
                    # Create prompt and classify
                    areas_text = "\n".join([f"{j+1}. {area.get('area', f'Area {j+1}')}" for j, area in enumerate(key_areas)])
                    feedback_text = "\n".join([f"Feedback {j+1}: {text}" for j, text in enumerate(chunk_texts)])
                    
                    prompt = f"""
                    I need to classify these customer feedback items into the most relevant categories.
                    
                    CATEGORIES:
                    {areas_text}
                    
                    FEEDBACK ITEMS:
                    {feedback_text}
                    
                    For each feedback item, tell me which categories it belongs to (multiple categories allowed).
                    Format as JSON:
                    {{
                        "classifications": [
                            {{
                                "feedback_index": 0,
                                "category_indices": [0, 2]
                            }},
                            ...
                        ]
                    }}
                    
                    Category indices are 0-based, corresponding to the numbered list above.
                    """
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": "You are a customer feedback classification system."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            response_format={"type": "json_object"}
                        )
                        
                        result = json.loads(response.choices[0].message.content)
                        classifications = []
                        
                        # Process classifications
                        if "classifications" in result:
                            for classification in result["classifications"]:
                                feedback_idx = classification.get("feedback_index", 0)
                                category_indices = classification.get("category_indices", [])
                                
                                if feedback_idx < len(chunk_texts):
                                    feedback_text = chunk_texts[feedback_idx]
                                    
                                    if not category_indices and key_areas:
                                        first_area = key_areas[0].get('area', 'Area 1')
                                        classifications.append((first_area, feedback_text))
                                    else:
                                        for cat_idx in category_indices:
                                            if 0 <= cat_idx < len(key_areas):
                                                area = key_areas[cat_idx].get('area', f'Area {cat_idx+1}')
                                                classifications.append((area, feedback_text))
                        return classifications
                    
                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {chunk_idx + 1}: {str(chunk_error)}")
                        # Assign all feedbacks in this chunk to the first category as fallback
                        if key_areas:
                            first_area = key_areas[0].get('area', 'Area 1')
                            return [(first_area, text) for text in chunk_texts]
                        return []
            
            # Prepare chunks
            chunks = []
            for i in range(0, len(all_feedback_items), chunk_size):
                chunk = all_feedback_items[i:i + min(chunk_size, len(all_feedback_items) - i)]
                chunks.append(chunk)
            
            # Process all chunks in parallel with rate limiting
            chunk_results = await asyncio.gather(
                *[process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
            )
            
            # Since we don't have embeddings, generate sentiment scores another way
            
            # Add all classified feedback to the result
            for classifications in chunk_results:
                for area, feedback_text in classifications:
                    classified_feedback[area].append(feedback_text)
            
            # Generate fallback sentiment scores
            # We'll use a simple keyword-based approach for fallback
            positive_keywords = ["good", "great", "excellent", "amazing", "love", "best", "happy", "perfect", "satisfied"]
            negative_keywords = ["bad", "poor", "terrible", "awful", "hate", "worst", "unhappy", "disappointed", "failure"]
            
            text_to_sentiment = {}
            for text in feedback_texts:
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_keywords if word in text_lower)
                neg_count = sum(1 for word in negative_keywords if word in text_lower)
                # Simple formula: (pos - neg) / (pos + neg + 1) to get score between -1 and 1
                sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1) if (pos_count + neg_count) > 0 else 0
                text_to_sentiment[text] = sentiment
        
        # Log classification stats
        logger.info("Classification results:")
        for area, feedbacks in classified_feedback.items():
            logger.info(f"  - {area}: {len(feedbacks)} items")
        
        # Map original data back to each feedback
        text_to_item = {item["text"]: item for item in all_feedback_items}
        enriched_feedback = {}
        
        for area, feedbacks in classified_feedback.items():
            enriched_feedback[area] = []
            for feedback_text in feedbacks:
                if feedback_text in text_to_item:
                    # Include original row data
                    original_item = text_to_item[feedback_text]
                    original_row = original_item.get("original_row", {"text": feedback_text})
                    
                    # Add sentiment score if available
                    if feedback_text in text_to_sentiment:
                        original_row["sentiment_score"] = text_to_sentiment[feedback_text]
                    else:
                        # Use a neutral sentiment as fallback
                        original_row["sentiment_score"] = 0.0
                        
                    enriched_feedback[area].append(original_row)
                else:
                    # Fallback if text not found in mapping
                    enriched_feedback[area].append({"text": feedback_text, "sentiment_score": 0.0})
        
        # Replace classified_feedback with enriched version
        classified_feedback = enriched_feedback

        return {
            "key_areas": key_areas,
            "classified_feedback": classified_feedback
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}\n{traceback.format_exc()}")
        # Fall back to raw data processing
        logger.info("Falling back to raw data processing")
        return await analyze_raw_data_chunks(client, csv_data)
async def format_analysis_results(analysis_results: Dict[str, Any], return_raw_feedback: bool = False) -> Dict[str, Any]:
    """
    Format the analysis results into the final structure.
    
    Args:
        analysis_results: Dictionary with key areas and classified feedback
        return_raw_feedback: Whether to include raw feedback text in the response
        
    Returns:
        Formatted results matching the required output structure
    """
    key_areas = analysis_results.get("key_areas", [])
    classified_feedback = analysis_results.get("classified_feedback", {})
    
    # Format the results
    formatted_results = []
    
    # Process each key area
    for area_obj in key_areas:
        if isinstance(area_obj, dict):
            # Get area name and problem from the dictionary
            area_name = area_obj.get("area", "Unknown Area")
            area_problem = area_obj.get("problem", f"Issues related to {area_name.lower()}")
        else:
            # Fallback for string-only areas
            area_name = str(area_obj)
            area_problem = f"Issues related to {area_name.lower()}"
        
        # Get the feedback for this area
        area_feedbacks = classified_feedback.get(area_name, [])
        
        result = {
            "key_area": area_name,
            "customer_problem": area_problem,
            "number_of_users": len(area_feedbacks)
        }
        
        # Only include raw feedback if requested
        if return_raw_feedback:
            result["raw_feedbacks"] = area_feedbacks
            
        formatted_results.append(result)
    
    # Sort by number of users (descending)
    formatted_results.sort(key=lambda x: x["number_of_users"], reverse=True)
    
    return {
        "analysis_results": formatted_results,
        "summary": {
            "total_feedback_items": sum(len(feedbacks) for feedbacks in classified_feedback.values()),
            "total_key_areas": len(key_areas)
        }
    }
@router.post("/analyze-feedback")
@async_timeout(120) 
async def analyze_feedback(
    file: UploadFile = File(...),
    return_raw_feedback: bool = Form(True),
    source: str = Form(None)  # Add optional source parameter
) -> JSONResponse:
    """
    Analyzes customer feedback from an uploaded CSV or Excel file.
    
    Args:
        file: Uploaded file (CSV or Excel)
        return_raw_feedback: Whether to include raw feedback text in the response
        source: Optional source value to override the Source column
        
    Returns:
        JSONResponse with analysis results
    """
    job_id = f"feedback_analysis_{int(time.time())}"
    
    try:
        start_time = time.time()
        logger.info(f"[JOB {job_id}] Starting analysis of file: {file.filename}")
        logger.info(f"[JOB {job_id}] Return raw feedback: {return_raw_feedback}")
        logger.info(f"[JOB {job_id}] Source override: {source if source else 'None'}")
        
        # Read the uploaded file
        logger.info(f"[JOB {job_id}] Reading file content")
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"[JOB {job_id}] File size: {file_size/1024:.1f} KB")
        
        # Determine file type based on extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Process Excel files
        if file_ext in ['.xlsx', '.xls', '.xlsm']:
            logger.info(f"[JOB {job_id}] Detected Excel file, processing with Excel-specific workflow")
            analysis_results = await process_excel_data(AZURE_CLIENT, file_content, file.filename, source)
        else:
            # Handle CSV and other file types with existing logic
            try:
                logger.info(f"[JOB {job_id}] Attempting UTF-8 decoding")
                file_text = file_content.decode("utf-8")
                logger.info(f"[JOB {job_id}] Successfully decoded with UTF-8")
            except UnicodeDecodeError:
                # Try different encoding if UTF-8 fails
                try:
                    logger.info(f"[JOB {job_id}] UTF-8 failed, attempting Latin-1 decoding")
                    file_text = file_content.decode("latin-1")
                    logger.info(f"[JOB {job_id}] Successfully decoded with Latin-1")
                except Exception:
                    # If all fails, save to temp file and use pandas to read
                    logger.info(f"[JOB {job_id}] Text decoding failed, using pandas for parsing")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
                        temp.write(file_content)
                        temp_path = temp.name
                        logger.info(f"[JOB {job_id}] Saved to temporary file: {temp_path}")
                    
                    try:
                        logger.info(f"[JOB {job_id}] Attempting to read with pandas")
                        df = pd.read_csv(temp_path)
                        file_text = df.to_csv(index=False)
                        logger.info(f"[JOB {job_id}] Successfully parsed with pandas: {df.shape[0]} rows, {df.shape[1]} columns")
                    except Exception as read_error:
                        logger.error(f"[JOB {job_id}] Pandas parsing failed: {str(read_error)}")
                        os.unlink(temp_path)
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Could not parse file: {str(read_error)}"
                        )
                    finally:
                        logger.info(f"[JOB {job_id}] Cleaning up temporary file")
                        os.unlink(temp_path)
            
            # Check if it looks like CSV
            is_csv = ',' in file_text and '\n' in file_text
            
            # Process the data
            if is_csv:
                logger.info(f"[JOB {job_id}] File detected as CSV, processing with structured data workflow")
                analysis_results = await process_csv_data(AZURE_CLIENT, file_text, source)
            else:
                logger.info(f"[JOB {job_id}] File not detected as CSV, processing as raw text")
                analysis_results = await analyze_raw_data_chunks(AZURE_CLIENT, file_text)
        
        # Format the results with the return_raw_feedback parameter
        logger.info(f"[JOB {job_id}] Formatting final analysis results (return_raw_feedback={return_raw_feedback})")
        formatted_results = await format_analysis_results(analysis_results, return_raw_feedback)
        
        
        # Log summary statistics
        areas_count = len(formatted_results.get("analysis_results", []))
        total_items = formatted_results.get("summary", {}).get("total_feedback_items", 0)
        
        # Log the detected key areas and their counts
        logger.info(f"[JOB {job_id}] ANALYSIS SUMMARY: {areas_count} key areas, {total_items} total feedback items")
        for area in formatted_results.get("analysis_results", []):
            logger.info(f"[JOB {job_id}] - {area['key_area']}: {area['number_of_users']} feedback items")
        
        elapsed_time = time.time() - start_time
        logger.info(f"[JOB {job_id}] Analysis completed in {elapsed_time:.2f} seconds")
        
        return JSONResponse(
            content=formatted_results,
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error analyzing feedback: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing feedback: {str(e)}"
        )
def integrate_with_main_app(app):
    """
    Integrates the customer feedback analysis router with a FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router, prefix="/feedback", tags=["customer_feedback"])
    logging.info("Customer feedback analysis API integrated with main app")

# Standalone app routes
@app.post("/analyze-feedback")
@async_timeout(120)
async def standalone_analyze_feedback(
    file: UploadFile = File(...),
    return_raw_feedback: bool = Form(False),
    source: str = Form(None)  # Add optional source parameter
) -> JSONResponse:
    """
    Standalone version of the analyze_feedback endpoint.
    
    Args:
        file: Uploaded file (expected to be CSV or Excel)
        return_raw_feedback: Whether to include raw feedback text in the response
        source: Optional source value to override the Source column
        
    Returns:
        JSONResponse with analysis results
    """
    return await analyze_feedback(file, return_raw_feedback, source)
@app.post("/classify-single-review")
async def standalone_classify_single_review(
    request: Request = None,
    review: str = Form(None),
    existing_json: str = Form(None)
) -> JSONResponse:
    """
    Standalone version of the classify-single-review endpoint.
    Accepts both JSON body and form data.
    
    Args:
        request: FastAPI request object containing JSON
        review: Review text (when using form data)
        existing_json: Existing categories (when using form data)
        
    Returns:
        JSONResponse with classification result containing key_area, customer_problem, and sentiment_score
    """
    # Handle both form data and JSON body
    if review is not None:
        # Form data was provided
        return await classify_single_review(review=review, existing_json=existing_json)
    else:
        # Process as JSON body
        return await classify_single_review(request=request)
        
@router.post("/classify-single-review")
async def classify_single_review(
    request: Request = None,
    review: str = None,
    existing_json: str = None
) -> JSONResponse:
    """
    Classifies a single review against existing categories using GPT-4.1.
    Accepts input either as a JSON request body or as form parameters.
    
    Args:
        request: FastAPI request object containing JSON with 'review' and 'existing_json'
        review: Direct string input for review (alternative to request)
        existing_json: Direct string/JSON input for categories (alternative to request)
        
    Returns:
        JSONResponse with classification result containing key_area, customer_problem, and sentiment_score
    """
    try:
        # Parse input from either request body or direct parameters
        review_text = ""
        existing_categories_raw = []
        
        # Get data from request body if provided and other params are None
        if request is not None and review is None:
            try:
                data = await request.json()
                review_text = data.get("review", "")
                existing_categories_raw = data.get("existing_json", [])
            except Exception as e:
                logger.warning(f"Error parsing request JSON: {str(e)}")
                # Try to get data from form if JSON parsing failed
                form = await request.form()
                review_text = form.get("review", "")
                existing_json_str = form.get("existing_json", "")
                if existing_json_str:
                    try:
                        existing_categories_raw = json.loads(existing_json_str)
                    except:
                        logger.warning("Could not parse existing_json as JSON")
                        existing_categories_raw = []
        else:
            # Get data from direct parameters
            review_text = review or ""
            if existing_json:
                try:
                    # Try to parse as JSON if it's a string
                    if isinstance(existing_json, str):
                        try:
                            existing_categories_raw = json.loads(existing_json)
                        except:
                            logger.warning("Could not parse existing_json string as JSON")
                            existing_categories_raw = []
                    else:
                        existing_categories_raw = existing_json
                except:
                    logger.warning("Error processing existing_json parameter")
                    existing_categories_raw = []
        
        # Store original raw request for potential fallback
        original_json_data = existing_json if isinstance(existing_json, str) else json.dumps(existing_categories_raw) if existing_categories_raw else "[]"
        
        # Validate review input - if empty, use a generic placeholder
        if not review_text or review_text.strip() == "":
            review_text = "This is a general feedback that needs classification."
            logger.warning("Empty review text provided, using generic placeholder")
        else:
            logger.info(f"Processing review: {review_text[:100]}...")
        
        # Process existing categories - handle empty input gracefully
        existing_categories = []
        
        # Handle different possible JSON formats
        if isinstance(existing_categories_raw, list):
            existing_categories = existing_categories_raw
        elif isinstance(existing_categories_raw, dict) and "analysis_results" in existing_categories_raw:
            existing_categories = existing_categories_raw.get("analysis_results", [])
        elif isinstance(existing_categories_raw, str) and existing_categories_raw.strip():
            try:
                json_data = json.loads(existing_categories_raw)
                if isinstance(json_data, list):
                    existing_categories = json_data
                elif isinstance(json_data, dict) and "analysis_results" in json_data:
                    existing_categories = json_data.get("analysis_results", [])
            except:
                logger.warning("Could not parse existing_json as JSON string")
        
        # If no categories provided or parsing failed, create a default category
        if not existing_categories:
            logger.warning("No valid categories found, using default categories")
        logger.info(f"Processing with {len(existing_categories)} categories")
        
        # Extract just key_area and customer_problem from existing categories
        simplified_categories = []
        for category in existing_categories:
            if isinstance(category, dict) and "key_area" in category and "customer_problem" in category:
                simplified_categories.append({
                    "key_area": category["key_area"],
                    "customer_problem": category["customer_problem"]
                })
        
        # If no valid categories found, create a default one
        if not simplified_categories:
            simplified_categories = {}        
        # Prepare list of existing categories for the prompt
        categories_list = "\n".join([
            f"{i+1}. Key Area: '{cat['key_area']}', Customer Problem: '{cat['customer_problem']}'"
            for i, cat in enumerate(simplified_categories)
        ])
        prompt = f"""
        # Customer Review Classification and Sentiment Analysis Task

        ## Your Objective
        1. Analyze the provided customer review and either:
            - Match it to an existing category if there's a good thematic fit (80%+ similarity)
            - Create a new category if the review doesn't clearly match any existing category
        2. Determine the sentiment score of the review on a scale from -1.0 (extremely negative) to 1.0 (extremely positive)

        For classification, provide:
            - `key_area`: A short (2-4 word) label identifying the general functional area
            - `customer_problem`: Concise summary of the specific issue (5-8 words)
            
        For sentiment, provide:
            - `sentiment_score`: A float value between -1.0 and 1.0 where:
                * -1.0 to -0.6: Extremely negative (angry, frustrated)
                * -0.6 to -0.2: Moderately negative (disappointed, dissatisfied)
                * -0.2 to 0.2: Neutral (factual, balanced)
                * 0.2 to 0.6: Moderately positive (pleased, satisfied)
                * 0.6 to 1.0: Extremely positive (delighted, enthusiastic)
            
        ## Classification Instructions
        CAREFULLY CONSIDER:
            1. ONLY match to an existing category when there's a CLEAR thematic match (the review discusses the same type of issue)
            2. CREATE A NEW CATEGORY when:
                - The review addresses a different functional area than existing categories
                - The review describes a different type of problem within a similar area
                - The review contains a substantially different complaint or suggestion

        ## Understanding the Database Structure
        The feedback database has these key fields in each row:
        - `key_area`: A short (2-4 word) label identifying the general topic area
        - `customer_problem`: Summary of the issue based on the EXACT REVIEW TEXT (upto 5-8 words)
        
        These two fields MUST ALWAYS BE PAIRED TOGETHER exactly as they appear in the existing categories.

        ## Existing Categories
        These are the EXACT `key_area` and `customer_problem` pairs currently in our database:
        {categories_list}

        ## Customer Review to Classify
        "{review_text}"

        ## Classification Instructions
        1. PRIORITIZE matching to an EXISTING category - this is CRITICAL for database consistency
        2. Look for thematic matches between the review content and the existing categories
        3. When you find a match, you MUST use BOTH the exact `key_area` AND `customer_problem` from that category (copy-paste exactly)
        4. Only create a new category if the review is unrelated to ALL existing categories

        ## Sentiment Analysis Instructions
        1. Analyze the emotional tone, language, and context of the review
        2. Consider both explicit sentiment words and implicit meaning
        3. Pay attention to intensity modifiers (very, extremely, somewhat)
        4. Account for mixed sentiments by determining the dominant tone
        5. Assign a precise score between -1.0 and 1.0, not just the range boundaries

        ## IMPORTANT REQUIREMENTS
        1. Your response MUST be a valid JSON object with EXACTLY these three fields:
           - key_area: The category name (preferably from existing categories)
           - customer_problem: The problem description (preferably from existing categories)
           - sentiment_score: A float value between -1.0 and 1.0
        2. DO NOT include any explanation, notes, or additional text
        3. DO NOT add any extra fields to the JSON
        4. COPY EXACT category names and descriptions when using existing categories

        ## EXAMPLES of Classification and Sentiment Scoring

        ### EXAMPLE 1: Negative Sentiment
        Review: "The app crashes every time I try to upload a photo of my receipt. I've tried multiple times and it's incredibly frustrating!"
        Correct Response:
        {{{{
            "key_area": "Mobile App Performance",
            "customer_problem": "App crashes during specific operations",
            "sentiment_score": -0.8
        }}}}
        Reasoning: High negativity (-0.8) due to frustration and repeated failure.

        ### EXAMPLE 2: Slightly Negative Sentiment
        Review: "Your website loads slowly on my browser sometimes. It's not the worst but could be better."
        Correct Response:
        {{{{
            "key_area": "Website Performance",
            "customer_problem": "Slow loading times and timeouts",
            "sentiment_score": -0.3
        }}}}
        Reasoning: Mildly negative (-0.3) with acknowledgment that it's "not the worst".

        ### EXAMPLE 3: Neutral Sentiment
        Review: "The mobile app has different features than the website. I can access my account on both but they look different."
        Correct Response:
        {{{{
            "key_area": "Cross-Platform Consistency",
            "customer_problem": "Features available on website missing from mobile app",
            "sentiment_score": 0.0
        }}}}
        Reasoning: Neutral (0.0) as it's a factual observation without clear emotion.

        ### EXAMPLE 4: Positive Sentiment
        Review: "I like how the app lets me track my orders easily. The notifications are helpful."
        Correct Response:
        {{{{
            "key_area": "App Functionality",
            "customer_problem": "Positive feedback about order tracking features",
            "sentiment_score": 0.6
        }}}}
        Reasoning: Moderately positive (0.6) with clear approval but not extreme enthusiasm.

        ## Response Format (STRICTLY FOLLOW THIS)
        ```json
        {{
            "key_area": "Exact category name",
            "customer_problem": "Describe the problem in 5 words or less",
            "sentiment_score": -0.7
        }}
        ```
        """
        
        # Store the original raw response for fallback
        raw_response = None
        
        # Retry mechanism - attempt up to 5 times to get valid JSON
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"GPT classification attempt {attempt+1}/{max_retries}")
                
                response = AZURE_CLIENT.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a customer feedback classification specialist who always responds with valid JSON. The json must contain key_area (headline of the issue), customer_problem (brief 7-8 word summary of the user problem), and sentiment_score (float from -1.0 to 1.0)."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                result_text = response.choices[0].message.content
                print(result_text)
                raw_response = result_text  # Store the raw response
                # Try to parse as JSON
                try:
                    result = json.loads(result_text)
                    
                    # Validate that the response contains the required fields
                    if "key_area" in result and "customer_problem" in result and "sentiment_score" in result:
                        # Check if this matches an existing category
                        is_existing_pair = False
                        for cat in simplified_categories:
                            if (cat["key_area"] == result["key_area"] and 
                                cat["customer_problem"] == result["customer_problem"]):
                                is_existing_pair = True
                                break
                        
                        # Ensure sentiment_score is a valid float between -1 and 1
                        sentiment_score = float(result["sentiment_score"])
                        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
                        
                        logger.info(f"Classification result: {result['key_area']} (existing pair: {is_existing_pair}), sentiment: {sentiment_score}")
                        return JSONResponse(content={
                            "key_area": result["key_area"],
                            "customer_problem": result["customer_problem"],
                            "sentiment_score": sentiment_score
                        })
                    else:
                        logger.warning(f"GPT response missing required fields: {result}")
                        # Continue to next attempt
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Invalid JSON on attempt {attempt+1}: {json_err}. Response: {result_text[:200]}...")
                    # Continue to next attempt
                    
            except Exception as e:
                logger.error(f"Error on attempt {attempt+1}: {str(e)}")
                # Continue to next attempt
            
            # Add slight delay between retries
            await asyncio.sleep(1)
        
        # If all attempts failed, try to extract from the raw response
        logger.warning("All GPT attempts failed, using fallback extraction")
        
        # Try to extract structured data from the raw response
        result = extract_classification_from_text(raw_response, simplified_categories, original_json_data)
        
        # Add a default sentiment score if not present
        if "sentiment_score" not in result:
            # Analyze text for basic sentiment as fallback
            sentiment_words = {
                "terrible": -0.9, "awful": -0.8, "horrible": -0.8, "bad": -0.6, 
                "poor": -0.5, "disappointing": -0.5, "issue": -0.4, "problem": -0.4,
                "not working": -0.6, "don't like": -0.5, "doesn't work": -0.6,
                "fails": -0.7, "failed": -0.7, "broken": -0.7, "useless": -0.8,
                "ok": 0.0, "okay": 0.0, "fine": 0.1, "works": 0.3,
                "good": 0.5, "nice": 0.6, "great": 0.7, "excellent": 0.8,
                "amazing": 0.9, "awesome": 0.9, "love": 0.8, "wonderful": 0.8
            }
            
            # Simple fallback sentiment analysis
            text_lower = review_text.lower()
            sentiments = []
            
            for word, score in sentiment_words.items():
                if word in text_lower:
                    sentiments.append(score)
            
            # Default to -0.1 for issues/problems if no sentiment words found
            sentiment_score = sum(sentiments) / len(sentiments) if sentiments else -0.1
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
            
            result["sentiment_score"] = sentiment_score
        
        logger.info(f"Fallback classification result with sentiment: {result}")
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error classifying single review: {str(e)}\n{traceback.format_exc()}")
        # Ultimate fallback - return a generic classification with neutral sentiment
        return JSONResponse(content={
            "key_area": "Miscellaneous Issue",
            "customer_problem": "Issues related to miscellaneous customer concerns",
            "sentiment_score": -0.1  # Slightly negative default for issues
        })

def extract_classification_from_text(text: str, existing_categories: List[Dict[str, str]], original_json: str = "[]") -> Dict[str, str]:
    """
    Fallback method to extract key_area, customer_problem, and sentiment_score from GPT response when JSON parsing fails.
    
    Args:
        text: Raw text response from GPT
        existing_categories: List of existing categories
        original_json: Original JSON input string
        
    Returns:
        Dictionary with key_area, customer_problem, and sentiment_score
    """
    # If no text provided, try to extract from original JSON or use default
    if not text or not text.strip():
        logger.warning("Empty response from GPT, attempting to use original data")
        try:
            # Try to extract from original JSON first
            if original_json and original_json.strip() not in ('[]', '{}', ''):
                try:
                    json_data = json.loads(original_json)
                    if isinstance(json_data, list) and len(json_data) > 0:
                        if "key_area" in json_data[0] and "customer_problem" in json_data[0]:
                            logger.info("Using first category from original JSON as fallback")
                            return {
                                "key_area": json_data[0]["key_area"],
                                "customer_problem": json_data[0]["customer_problem"],
                                "sentiment_score": 0.0  # Default neutral sentiment
                            }
                except:
                    pass
            
            # If original JSON parsing fails or is empty, use the first existing category
            if existing_categories:
                return {
                    "key_area": existing_categories[0]["key_area"],
                    "customer_problem": existing_categories[0]["customer_problem"],
                    "sentiment_score": 0.0  # Default neutral sentiment
                }
        except:
            pass
        
        # Final fallback
        return {
            "key_area": "Miscellaneous Issue",
            "customer_problem": "Issues related to miscellaneous customer concerns",
            "sentiment_score": -0.1  # Slightly negative default for issues
        }
    
    try:
        # Extract sentiment score from text if present
        sentiment_pattern = r'"sentiment_score"\s*:\s*(-?\d+\.?\d*)'
        sentiment_match = re.search(sentiment_pattern, text)
        sentiment_score = 0.0  # Default neutral
        
        if sentiment_match:
            try:
                sentiment_score = float(sentiment_match.group(1))
                sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
            except (ValueError, TypeError):
                # Invalid float, use default
                sentiment_score = 0.0
        
        # Multi-tiered parsing approach
        
        # 1. Try with the raw text as is
        try:
            raw_json = json.loads(text)
            if "key_area" in raw_json and "customer_problem" in raw_json:
                # Add sentiment if present or use the extracted one
                if "sentiment_score" in raw_json:
                    try:
                        sentiment_score = float(raw_json["sentiment_score"])
                        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
                    except (ValueError, TypeError):
                        pass  # Keep the previously extracted sentiment
                
                return {
                    "key_area": raw_json["key_area"],
                    "customer_problem": raw_json["customer_problem"],
                    "sentiment_score": sentiment_score
                }
        except:
            pass
        
        # 2. Clean and try again
        cleaned_text = text.strip()
        # Remove markdown code block markers and extra spaces
        cleaned_text = re.sub(r'```json|```|`', '', cleaned_text).strip()
        try:
            cleaned_json = json.loads(cleaned_text)
            if "key_area" in cleaned_json and "customer_problem" in cleaned_json:
                # Add sentiment if present or use the extracted one
                if "sentiment_score" in cleaned_json:
                    try:
                        sentiment_score = float(cleaned_json["sentiment_score"])
                        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
                    except (ValueError, TypeError):
                        pass  # Keep the previously extracted sentiment
                
                return {
                    "key_area": cleaned_json["key_area"],
                    "customer_problem": cleaned_json["customer_problem"],
                    "sentiment_score": sentiment_score
                }
        except:
            pass
        
        # 3. Try to extract any JSON-like content
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, cleaned_text)
        
        if json_matches:
            for json_str in json_matches:
                try:
                    # Try to parse this JSON fragment
                    result = json.loads(json_str)
                    if "key_area" in result and "customer_problem" in result:
                        # Add sentiment if present or use the extracted one
                        if "sentiment_score" in result:
                            try:
                                sentiment_score = float(result["sentiment_score"])
                                sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1.0, 1.0]
                            except (ValueError, TypeError):
                                pass  # Keep the previously extracted sentiment
                        
                        # Check if this is an existing pair
                        for cat in existing_categories:
                            if (cat["key_area"] == result["key_area"] and 
                                cat["customer_problem"] == result["customer_problem"]):
                                return {
                                    "key_area": cat["key_area"],
                                    "customer_problem": cat["customer_problem"],
                                    "sentiment_score": sentiment_score
                                }
                        # If not an exact match, still return the extracted pair
                        return {
                            "key_area": result["key_area"],
                            "customer_problem": result["customer_problem"],
                            "sentiment_score": sentiment_score
                        }
                except:
                    continue
        
        # 4. Try to extract key_area and customer_problem from text using regex
        key_area_pattern = r'"key_area"\s*:\s*"([^"]+)"'
        problem_pattern = r'"customer_problem"\s*:\s*"([^"]+)"'
        
        key_area_match = re.search(key_area_pattern, cleaned_text)
        problem_match = re.search(problem_pattern, cleaned_text)
        
        if key_area_match and problem_match:
            key_area = key_area_match.group(1)
            customer_problem = problem_match.group(1)
            
            # Check if this is an existing pair
            for cat in existing_categories:
                if (cat["key_area"] == key_area and cat["customer_problem"] == customer_problem):
                    return {
                        "key_area": cat["key_area"],
                        "customer_problem": cat["customer_problem"],
                        "sentiment_score": sentiment_score
                    }
            
            # If not an exact match, return the extracted values
            return {
                "key_area": key_area,
                "customer_problem": customer_problem,
                "sentiment_score": sentiment_score
            }
        
        # 5. Look for exact matches with existing categories in the text
        for category in existing_categories:
            # If both key_area and customer_problem appear in the text, prioritize that match
            if category["key_area"] in cleaned_text and category["customer_problem"] in cleaned_text:
                logger.info(f"Found exact category match in text: {category['key_area']}")
                return {
                    "key_area": category["key_area"],
                    "customer_problem": category["customer_problem"],
                    "sentiment_score": sentiment_score
                }
        
        # 6. Look for partial matches with existing categories
        for category in existing_categories:
            # Look for substantial parts of the customer_problem in the text
            problem_parts = category["customer_problem"].split()
            if len(problem_parts) >= 3:
                # Check if at least 3 consecutive words from the problem appear in the text
                for i in range(len(problem_parts) - 2):
                    phrase = " ".join(problem_parts[i:i+3])
                    if phrase.lower() in cleaned_text.lower():
                        logger.info(f"Found partial match on customer problem: {category['key_area']}")
                        return {
                            "key_area": category["key_area"],
                            "customer_problem": category["customer_problem"],
                            "sentiment_score": sentiment_score
                        }
        
        # 7. Check for key_area matches
        for category in existing_categories:
            if category["key_area"].lower() in cleaned_text.lower():
                logger.info(f"Found key_area match: {category['key_area']}")
                return {
                    "key_area": category["key_area"],
                    "customer_problem": category["customer_problem"],
                    "sentiment_score": sentiment_score
                }
        
        # 8. If all else fails, try to return raw JSON as last resort
        try:
            # If text looks like JSON but doesn't have the expected fields
            if text.strip().startswith('{') and text.strip().endswith('}'):
                raw_json_attempt = json.loads(text.strip())
                # If it's valid JSON with any fields, add the required fields with defaults
                if isinstance(raw_json_attempt, dict):
                    result = {
                        "key_area": raw_json_attempt.get("key_area", "Miscellaneous Issue"),
                        "customer_problem": raw_json_attempt.get("customer_problem", "General customer feedback"),
                        "sentiment_score": sentiment_score
                    }
                    return result
        except:
            pass
        
        # 9. Final fallback - return first category
        if existing_categories:
            logger.info("Using first existing category as final fallback")
            return {
                "key_area": existing_categories[0]["key_area"],
                "customer_problem": existing_categories[0]["customer_problem"],
                "sentiment_score": sentiment_score
            }
        
        # 10. Absolute last resort
        return {
            "key_area": "Miscellaneous Issue",
            "customer_problem": "Issues related to miscellaneous customer concerns",
            "sentiment_score": sentiment_score
        }
    
    except Exception as e:
        logger.error(f"Error in fallback extraction: {str(e)}")
        # Return the original JSON if all else fails
        try:
            if original_json and original_json not in ('[]', '{}', ''):
                json_data = json.loads(original_json)
                if isinstance(json_data, list) and len(json_data) > 0:
                    return {
                        "key_area": json_data[0].get("key_area", "Miscellaneous Issue"),
                        "customer_problem": json_data[0].get("customer_problem", "General customer feedback"),
                        "sentiment_score": -0.1  # Slightly negative default for issues
                    }
        except:
            pass
            
        # Absolute final fallback
        return {
            "key_area": "Miscellaneous Issue",
            "customer_problem": "Issues related to miscellaneous customer concerns",
            "sentiment_score": -0.1  # Slightly negative default for issues
        }
# Main function for running as standalone server
if __name__ == "__main__":
    import uvicorn
    
    # Set up server information
    port = int(os.environ.get("PORT", 8081))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"┌───────────────────────────────────────────────────────┐")
    print(f"│                                                       │")
    print(f"│  Customer Feedback Analysis API Server                │")
    print(f"│                                                       │")
    print(f"│  Endpoint:     http://{host}:{port}/analyze-feedback  │")
    print(f"│  Documentation: http://{host}:{port}/docs             │")
    print(f"│                                                       │")
    print(f"│  Logs:         customer_feedback.log                  │")
    print(f"│                                                       │")
    print(f"└───────────────────────────────────────────────────────┘")
    
    # Configure server settings
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
