import logging
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import time
import asyncio
from io import StringIO
import tempfile
from openai import AzureOpenAI
from fuzzywuzzy import process

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
AZURE_ENDPOINT = "https://kb-stellar.openai.azure.com/"
AZURE_API_KEY = "bc0ba854d3644d7998a5034af62d03ce" 
AZURE_API_VERSION = "2024-02-01"  # Updated API version

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-large"  # Correct embedding model name

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

# Router is kept for optional integration scenarios
router = APIRouter()

# Common key areas from screenshot and expanded options
COMMON_KEY_AREAS = [
    {"area": "Reporting & Analytics", "problem": "Difficulty generating needed reports for data-driven decisions"},
    {"area": "Automation & Workflows", "problem": "Manual processes that are tedious and error-prone"},
    {"area": "Mobile Experience", "problem": "Limited mobile app functionality making remote work difficult"},
    {"area": "Customization & Configurability", "problem": "Lack of flexibility in platform configuration"},
    {"area": "Integrations & API Access", "problem": "Limited integrations slowing down workflows"},
    {"area": "Access Control & Permissions", "problem": "Inadequate granular permissions for different teams"},
    {"area": "Collaboration & Multi-Team Support", "problem": "Difficulties managing multiple teams and collaboration"},
    {"area": "AI-Powered Decision Support", "problem": "Need help prioritizing features and making decisions"}
]

# Extended key areas for broader coverage
EXTENDED_KEY_AREAS = COMMON_KEY_AREAS + [
    {"area": "User Interface", "problem": "Confusing or complicated user interface design"},
    {"area": "Performance & Speed", "problem": "System slowness or performance issues"},
    {"area": "Data Visualization", "problem": "Limited or ineffective ways to visualize data"},
    {"area": "Onboarding & Training", "problem": "Difficulty learning how to use the platform effectively"},
    {"area": "Customer Support", "problem": "Issues with getting help or support when needed"},
    {"area": "Pricing & Cost", "problem": "Concerns about pricing structure or overall cost"},
    {"area": "Feature Requests", "problem": "Requests for specific new functionality"},
    {"area": "Bugs & Stability", "problem": "Unexpected errors or system crashes"},
    {"area": "Documentation", "problem": "Unclear or insufficient documentation"},
    {"area": "Reliability", "problem": "System downtime or reliability concerns"},
    {"area": "Search Functionality", "problem": "Difficulty finding information or content"},
    {"area": "Data Management", "problem": "Challenges with importing, exporting, or managing data"},
    {"area": "Security & Privacy", "problem": "Concerns about data security or privacy"},
    {"area": "Cross-platform Compatibility", "problem": "Issues with different devices or browsers"},
    {"area": "Accessibility", "problem": "Challenges for users with disabilities"},
    {"area": "Notifications & Alerts", "problem": "Missing or ineffective notification systems"},
    {"area": "User Roles", "problem": "Limitations in defining different user roles and capabilities"}
]

def create_client():
    """Creates an AzureOpenAI client instance."""
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

async def identify_key_areas(client: AzureOpenAI, data_sample: str, max_areas: int = 8) -> List[Dict[str, str]]:
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
        
        # Prepare the list of common areas for the model to consider
        common_areas_json = json.dumps(COMMON_KEY_AREAS, indent=2)
        extended_areas_json = json.dumps(EXTENDED_KEY_AREAS, indent=2)
        
        # Prepare the prompt for OpenAI
        prompt = f"""
        # Customer Feedback Analysis Task

        You are a senior product insights analyst specializing in customer feedback pattern recognition. Your expertise lies in identifying underlying themes and problems from diverse customer feedback.

        ## Your Objective
        Analyze the provided customer feedback data and identify the most significant problem areas that customers are experiencing. These insights will directly inform product development priorities.

        ## Customer Feedback Data Sample
        ```
        {data_sample}
        ```

        ## Reference Categories
        Below are common problem areas we've seen in similar products. Use these as references but don't be limited by them:

        ### Common Problem Areas:
        {common_areas_json}

        ### Extended Problem Areas:
        {extended_areas_json}

        ## Analysis Instructions:
        1. Carefully read and understand all customer feedback in the sample
        2. Identify the top {max_areas} most significant problem areas based on:
           - Frequency (how many customers mention this issue)
           - Severity (how impactful the problem seems to be)
           - Specificity (clear, actionable problem statements)
           - Business impact (issues affecting core product value)
        
        3. For each identified problem area:
           - Create a concise, descriptive title (2-5 words) reflecting the functional area
           - Write a specific problem statement from the customer's perspective
           - Ensure the problem is concrete enough to be actionable
           - Capture the essence of what customers are struggling with

        4. You may:
           - Select directly from the reference categories if they match well
           - Adapt a reference category with more specific wording
           - Create entirely new categories if the existing ones don't capture the feedback
           - Combine similar issues into a single coherent problem area

        ## Response Format Requirements
        You must format your response as a JSON array with each problem area having 'area' and 'problem' keys:

        ```json
        [
            {{"area": "Reporting & Analytics", "problem": "I can't generate the reports I need to make data-driven decisions."}},
            {{"area": "Mobile Experience", "problem": "The mobile app lacks key features, making it hard to work on the go."}}
        ]
        ```

        Think carefully about naming each area - these need to be specific enough to be meaningful but general enough to group similar issues.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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

async def get_embeddings(client: AzureOpenAI, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        client: Azure OpenAI client
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    try:
        # Process in batches to avoid request size limits
        batch_size = 100
        all_embeddings = []
        
        # Hardcoded configuration for embeddings
        EMBEDDING_MODEL = "text-embedding-3-large"  # Updated to correct model name
        EMBEDDING_KEY = "bc0ba854d3644d7998a5034af62d03ce"  # Hardcoded key
        
        # Create a dedicated client for embeddings to ensure correct configuration
        embedding_client = AzureOpenAI(
            api_version="2024-02-01",  # Updated API version
            azure_endpoint=AZURE_ENDPOINT,
            api_key=EMBEDDING_KEY
        )
        
        logger.info(f"Generating embeddings for {len(texts)} texts using model {EMBEDDING_MODEL}")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = embedding_client.embeddings.create(
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
                dim = 3072  # text-embedding-3-large dimension
                logger.warning(f"Using fallback random embeddings for batch {i//batch_size + 1} with dimension {dim}")
                for _ in range(len(batch_texts)):
                    random_embedding = list(np.random.normal(0, 0.1, dim))
                    all_embeddings.append(random_embedding)
        
        return all_embeddings
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Generate random embeddings as fallback
        logger.warning("Using fallback random embeddings for all texts")
        dim = 3072  # text-embedding-3-large dimension
        return [list(np.random.normal(0, 0.1, dim)) for _ in range(len(texts))]

async def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

async def identify_relevant_columns(client: AzureOpenAI, df: pd.DataFrame) -> Dict[str, str]:
    """
    Uses OpenAI to identify which columns contain ratings and feedback text.
    
    Args:
        client: Azure OpenAI client
        df: Pandas DataFrame containing the data
        
    Returns:
        Dictionary with 'rating_col' and 'feedback_col' keys
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
        Carefully examine the dataset and identify exactly two types of columns:

        1. CUSTOMER FEEDBACK COLUMN:
           - Contains textual customer opinions, comments, reviews, or feedback
           - Usually includes sentences, paragraphs, or detailed comments
           - May contain terms like "liked", "disliked", "issue with", "problem", etc.
           - Look for the column with the most detailed textual content
        
        2. RATING COLUMN:
           - Contains numerical scores (1-5, 1-10) or textual ratings ("Excellent", "Poor")
           - Often named "rating", "score", "stars", "satisfaction", etc.
           - May be presented as numbers or categorical values

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
            "rating_col": "1: [Exact Column Name]"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        columns = {"feedback_col": None, "rating_col": None}
        
        for col_type in ["feedback_col", "rating_col"]:
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
        
        # Fallbacks if columns not identified
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
        
        logger.info(f"COLUMN DETECTION COMPLETE: feedback={columns['feedback_col']}, rating={columns['rating_col']}")
        
        # Log column statistics for verification
        if columns["feedback_col"]:
            feedback_col = columns["feedback_col"]
            non_null_count = df[feedback_col].notna().sum()
            avg_length = df[feedback_col].astype(str).str.len().mean()
            logger.info(f"Feedback column '{feedback_col}' stats: {non_null_count} non-null values, avg length: {avg_length:.1f} chars")
        
        if columns["rating_col"]:
            rating_col = columns["rating_col"]
            value_counts = df[rating_col].value_counts().to_dict()
            logger.info(f"Rating column '{rating_col}' value distribution: {value_counts}")
        
        return columns
        
    except Exception as e:
        logging.error(f"Error identifying columns: {str(e)}")
        # Return empty result, will fall back to heuristics
        return {"feedback_col": None, "rating_col": None}

async def classify_feedback_with_openai(
    client: AzureOpenAI,
    key_areas: List[Dict[str, str]],
    feedback_chunk: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Classifies a chunk of feedback items into key areas using OpenAI directly.
    
    Args:
        client: Azure OpenAI client
        key_areas: List of key area dictionaries
        feedback_chunk: List of feedback dictionaries with 'text' and 'index' keys
        
    Returns:
        List of classified feedback with area assignments
    """
    try:
        logger.info(f"Classifying chunk of {len(feedback_chunk)} feedback items using OpenAI")
        
        # Format key areas for the prompt
        areas_text = "\n".join([f"{i+1}. {area['area']}: {area['problem']}" for i, area in enumerate(key_areas)])
        
        # Format feedback for the prompt
        feedback_text = "\n".join([f"Feedback {i+1}: {item['text']}" for i, item in enumerate(feedback_chunk)])
        
        prompt = f"""
        # Customer Feedback Classification Task

        ## Problem Context
        You are a specialized customer feedback analyst with expertise in categorizing feedback into relevant problem areas. This classification will be used to prioritize product improvements and understand customer pain points.

        ## Key Problem Areas Identified:
        ```
        {areas_text}
        ```

        ## Customer Feedback To Classify:
        ```
        {feedback_text}
        ```

        ## Your Classification Task

        Analyze each feedback item carefully and determine which problem area(s) it belongs to. Apply these principles:

        1. **Multi-area Classification**: A single feedback may address multiple problem areas. Identify ALL relevant areas.
        2. **Semantic Understanding**: Look for both explicit mentions and implicit references to problem areas.
        3. **Customer Intent**: Focus on the underlying customer need or frustration.
        4. **Contextual Relevance**: Consider the business context and how the feedback relates to product functionality.
        5. **Evidence-Based**: Base your classification on specific phrases or sentiments in the feedback.

        ### Classification Guidelines:
        - Assign MULTIPLE areas if the feedback touches on different issues
        - If a feedback is vague but leans toward a specific area, assign it there
        - For ambiguous feedback that could fit multiple areas, assign to the most relevant ones
        - If a feedback doesn't clearly match ANY area, leave the areas array empty
        - Don't force-fit feedback into areas - quality of classification is critical

        ## Required Response Format
        Respond with a JSON object containing classifications for each feedback item:

        ```json
        {{
            "classifications": [
                {{
                    "feedback_index": 0,
                    "areas": [0, 2],
                    "justification": "This feedback mentions reporting issues and data visualization problems"
                }},
                {{
                    "feedback_index": 1,
                    "areas": [1],
                    "justification": "Clear mention of mobile app limitations"
                }},
                ...
            ]
        }}
        ```

        - "feedback_index" must match the index in the provided feedback list (0-based)
        - "areas" must contain indices of matching problem areas (0-based)
        - Include a brief justification explaining your classification reasoning
        - If no areas match, use an empty array: "areas": []

        Think carefully about each piece of feedback and its relationship to the defined problem areas.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a customer feedback classification system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        classifications = result.get("classifications", [])
        
        # Log classification details
        logger.info(f"Received {len(classifications)} classifications from OpenAI")
        for c in classifications[:3]:  # Log a few examples for debugging
            justification = c.get("justification", "No justification provided")
            area_indices = c.get("areas", [])
            area_names = [key_areas[idx]["area"] for idx in area_indices if 0 <= idx < len(key_areas)]
            logger.info(f"  - Feedback {c.get('feedback_index')}: assigned to {area_names} - {justification}")
        
        # Process the classifications
        classified_feedback = []
        
        for item in feedback_chunk:
            matching_classification = next(
                (c for c in classifications if c.get("feedback_index") == item.get("chunk_index", 0)), 
                {"areas": []}
            )
            
            area_indices = matching_classification.get("areas", [])
            matched_areas = [key_areas[idx]["area"] for idx in area_indices if 0 <= idx < len(key_areas)]
            justification = matching_classification.get("justification", "")
            
            # If no areas matched and we have areas, assign to most likely area based on content
            if not matched_areas and key_areas:
                logger.warning(f"No areas matched for feedback: '{item['text'][:100]}...' - using fallback")
                # Simple heuristic: use the first area as default
                matched_areas = [key_areas[0]["area"]]
            
            classified_feedback.append({
                "feedback": item["text"],
                "original_index": item["original_index"],
                "areas": matched_areas,
                "justification": justification
            })
            
        # Log overall statistics
        area_counts = {}
        for item in classified_feedback:
            for area in item["areas"]:
                area_counts[area] = area_counts.get(area, 0) + 1
                
        logger.info("Classification distribution:")
        for area, count in area_counts.items():
            logger.info(f"  - {area}: {count} items")
            
        logger.info(f"Successfully classified {len(classified_feedback)} feedback items")
        
        return classified_feedback
        
    except Exception as e:
        logging.error(f"Error classifying feedback with OpenAI: {str(e)}")
        # Fallback: assign all feedback to first area or "General Issues"
        default_area = key_areas[0]["area"] if key_areas else "General Issues"
        return [
            {
                "feedback": item["text"],
                "original_index": item["original_index"],
                "areas": [default_area]
            }
            for item in feedback_chunk
        ]

async def classify_feedback_with_embeddings(
    client: AzureOpenAI, 
    feedback_items: List[Dict[str, Any]], 
    key_areas: List[Dict[str, str]],
    similarity_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Classifies feedback items into key areas using embeddings.
    
    Args:
        client: Azure OpenAI client
        feedback_items: List of feedback dictionaries with 'text' and 'original_index' keys
        key_areas: List of key area dictionaries
        similarity_threshold: Minimum similarity score to classify a feedback
        
    Returns:
        List of classified feedback with area assignments
    """
    try:
        # Create texts to embed for key areas
        area_texts = [f"{area['area']}: {area['problem']}" for area in key_areas]
        
        # Get texts to embed for feedback
        feedback_texts = [item["text"] for item in feedback_items]
        
        if not feedback_texts or not area_texts:
            return []
        
        # Get embeddings
        logging.info(f"Generating embeddings for {len(area_texts)} key areas and {len(feedback_texts)} feedback items")
        area_embeddings = await get_embeddings(client, area_texts)
        feedback_embeddings = await get_embeddings(client, feedback_texts)
        
        # Classify each feedback item
        classified_feedback = []
        
        for i, feedback_embedding in enumerate(feedback_embeddings):
            feedback_item = feedback_items[i]
            
            # Calculate similarity with each key area
            similarities = []
            for j, area_embedding in enumerate(area_embeddings):
                similarity = await cosine_similarity(feedback_embedding, area_embedding)
                similarities.append((j, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Assign to all areas that meet the threshold
            matched_areas = []
            for j, similarity in similarities:
                if similarity >= similarity_threshold:
                    area_name = key_areas[j]['area']
                    matched_areas.append(area_name)
            
            # If not assigned to any area, assign to the most similar one
            if not matched_areas and similarities:
                best_match_idx = similarities[0][0]
                area_name = key_areas[best_match_idx]['area']
                matched_areas.append(area_name)
            
            classified_feedback.append({
                "feedback": feedback_item["text"],
                "original_index": feedback_item["original_index"],
                "areas": matched_areas
            })
        
        return classified_feedback
        
    except Exception as e:
        logging.error(f"Error classifying feedback with embeddings: {str(e)}")
        # Fallback: assign all feedback to first area or "General Issues"
        default_area = key_areas[0]["area"] if key_areas else "General Issues"
        return [
            {
                "feedback": item["text"],
                "original_index": item["original_index"],
                "areas": [default_area]
            }
            for item in feedback_items
        ]

async def analyze_raw_data_chunks(
    client: AzureOpenAI, 
    raw_data: str,
    chunk_size: int = 8000,
    overlap: int = 500
) -> Dict[str, Any]:
    """
    Analyzes raw unstructured data by chunking and processing with OpenAI.
    
    Args:
        client: Azure OpenAI client
        raw_data: Raw text data
        chunk_size: Size of chunks to process
        overlap: Overlap between chunks
        
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
                    model="gpt-4o-mini",
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
                    areas = await identify_key_areas(client, sample_feedback, max_areas=5)
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
            Please consolidate these into 5-8 main categories:
            
            {areas_text}
            
            For each category:
            1. Provide a short, consistent name (2-3 words)
            
            Only include the category names, no descriptions or problems.
            Format your response as a JSON array of strings.
            Example: ["Category 1", "Category 2", "Category 3"]
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
                    if isinstance(v, list) and all(isinstance(x, str) for x in v):
                        final_areas = v
                        break
            elif isinstance(result, list) and all(isinstance(x, str) for x in result):
                final_areas = result
            else:
                # Use unique categories from chunks
                final_areas = list(set(chunk_areas)) if all(isinstance(x, str) for x in chunk_areas) else ["General Issues", "Equipment Problems", "Staff & Service", "Facility Maintenance"]
        else:
            final_areas = ["General Issues", "Equipment Problems", "Staff & Service", "Facility Maintenance"]
        
        logger.info(f"Consolidated into {len(final_areas)} final areas: {final_areas}")
            
        # STAGE 3: Use embeddings for fast classification
        # Generate embeddings for key areas
        key_area_embeddings = await get_embeddings(client, final_areas)
        
        # Generate embeddings for all feedback
        feedback_texts = [item["text"] for item in all_feedbacks]
        
        # Process in batches to handle potential size limitations
        batch_size = 300
        all_feedback_embeddings = []
        
        for i in range(0, len(feedback_texts), batch_size):
            batch = feedback_texts[i:i + batch_size]
            logger.info(f"Generating embeddings for feedback batch {i//batch_size + 1}/{(len(feedback_texts) + batch_size - 1)//batch_size}")
            batch_embeddings = await get_embeddings(client, batch)
            all_feedback_embeddings.extend(batch_embeddings)
        
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
        classified_feedback = {str(area): [] for area in final_areas}
        
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
                    area = str(final_areas[match_idx])
                    classified_feedback[area].append(feedback_texts[i])
        
        logger.info(f"Classified {len(all_feedbacks)} feedback items into {len(final_areas)} areas")
        
        return {
            "key_areas": final_areas,
            "classified_feedback": classified_feedback
        }
        
    except Exception as e:
        logger.error(f"Error in raw data analysis: {str(e)}\n{traceback.format_exc()}")
        basic_areas = ["General Issues", "Equipment Problems", "Staff & Service", "Facility Maintenance"]
        return {
            "key_areas": basic_areas,
            "classified_feedback": {str(area): [item["text"] for item in all_feedbacks[:len(all_feedbacks)//4]] 
                                  for i, area in enumerate(basic_areas)}
        }

async def process_csv_data(
    client: AzureOpenAI, 
    csv_data: str
) -> Dict[str, Any]:
    """
    Process structured CSV data to extract and classify feedback.
    
    Args:
        client: Azure OpenAI client
        csv_data: CSV data as string
        
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
        columns = await identify_relevant_columns(client, df)
        feedback_col = columns.get("feedback_col")
        rating_col = columns.get("rating_col")
        
        logger.info(f"Using columns: feedback={feedback_col}, rating={rating_col}")
        
        # Extract feedback items, handling NaN values
        if feedback_col and feedback_col in df.columns:
            # Skip 5-star reviews if rating column exists
            if rating_col and rating_col in df.columns:
                # Convert ratings to numeric if possible
                try:
                    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
                except:
                    logger.warning(f"Could not convert ratings to numeric in column {rating_col}")
                
                # Count reviews by rating for logging
                rating_counts = df[rating_col].value_counts().to_dict()
                logger.info(f"Rating distribution: {rating_counts}")
                
                # Filter out 5-star reviews
                try:
                    high_rating_mask = df[rating_col] >= 5
                    five_star_count = high_rating_mask.sum()
                    logger.info(f"Skipping {five_star_count} high-rated reviews (5-star)")
                    df = df[~high_rating_mask]
                except:
                    logger.warning("Could not filter out 5-star reviews, processing all")
            
            # Extract feedback with ratings
            feedback_items = []
            for idx, row in df.iterrows():
                if pd.notna(row[feedback_col]) and str(row[feedback_col]).strip():
                    # Include rating if available
                    feedback_text = str(row[feedback_col])
                    if rating_col and rating_col in df.columns and pd.notna(row[rating_col]):
                        feedback_text = f"[Rating: {row[rating_col]}] {feedback_text}"
                    
                    feedback_items.append({
                        "text": feedback_text,
                        "original_index": idx,
                        "original_row": row.to_dict()
                    })
        else:
            # If no feedback column identified, use the first text column
            for col in df.columns:
                if df[col].dtype == 'object':
                    feedback_items = []
                    for idx, row in df.iterrows():
                        if pd.notna(row[col]) and str(row[col]).strip():
                            feedback_items.append({
                                "text": str(row[col]),
                                "original_index": idx,
                                "original_row": row.to_dict()
                            })
                    break
        
        if not feedback_items:
            raise ValueError("No valid feedback found in the data")
            
        logger.info(f"Extracted {len(feedback_items)} feedback items")
            
        # Sample for key area identification (limit to 100 items to avoid token limits)
        sample_feedbacks = [item["text"] for item in feedback_items[:100]]
        sample_feedback_text = '\n'.join(sample_feedbacks)
        
        # Identify key areas - make sure we get string keys
        key_area_list = await identify_key_areas(client, sample_feedback_text)
        logger.info(f"Identified {len(key_area_list)} key areas before type conversion")
        
        # Ensure key_areas are strings, not dictionaries
        # Check if we got a list of dictionaries instead of strings
        if key_area_list and isinstance(key_area_list[0], dict):
            # Extract the 'area' field if present
            key_areas = [area.get('area', f"Area {i+1}") for i, area in enumerate(key_area_list)]
            logger.info(f"Converted dictionary key areas to strings: {key_areas}")
        else:
            # Already in correct format
            key_areas = key_area_list
            
        logger.info(f"Final key areas (as strings): {key_areas}")
        
        # Initialize the result structure with empty lists
        classified_feedback = {str(area): [] for area in key_areas}
        logger.info(f"Initialized classified_feedback with {len(classified_feedback)} area keys")
        
        # Try embeddings for classification first
        try:
            logger.info("Using embeddings for fast feedback classification")
            
            # Step 1: Generate embeddings for key areas
            key_area_embeddings = await get_embeddings(client, key_areas)
            logger.info(f"Generated embeddings for {len(key_areas)} key areas")
            
            # Step 2: Generate embeddings for all feedback in one batch
            feedback_texts = [item["text"] for item in feedback_items]
            
            # Process in batches to handle potential size limitations
            batch_size = 300
            all_feedback_embeddings = []
            
            for i in range(0, len(feedback_texts), batch_size):
                batch = feedback_texts[i:i + batch_size]
                logger.info(f"Generating embeddings for feedback batch {i//batch_size + 1}/{(len(feedback_texts) + batch_size - 1)//batch_size}")
                batch_embeddings = await get_embeddings(client, batch)
                all_feedback_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated embeddings for {len(all_feedback_embeddings)} feedback items")
            
            # Step 3: Calculate similarity and classify
            # For faster processing, use numpy operations
            key_area_matrix = np.array(key_area_embeddings)
            feedback_matrix = np.array(all_feedback_embeddings)
            
            # Verify embedding dimensions match
            key_area_dim = key_area_matrix.shape[1]
            feedback_dim = feedback_matrix.shape[1]
            
            logger.info(f"Embedding dimensions - Key areas: {key_area_matrix.shape}, Feedback: {feedback_matrix.shape}")
            
            if key_area_dim != feedback_dim:
                logger.error(f"Embedding dimensions mismatch: key_areas({key_area_dim}) != feedback({feedback_dim})")
                raise ValueError(f"Embedding dimensions don't match: {key_area_dim} vs {feedback_dim}")
            
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
            similarity_matrix = cosine_similarity_matrix(feedback_matrix, key_area_matrix)
            
            # Classify feedback based on similarity
            similarity_threshold = 0.25  # Even lower threshold for better matching
            
            matched_count = 0
            for i, similarities in enumerate(similarity_matrix):
                # Get indices where similarity exceeds threshold
                matches = np.where(similarities > similarity_threshold)[0]
                
                # If no matches, use the best match
                if len(matches) == 0:
                    best_match = np.argmax(similarities)
                    matches = [best_match]
                
                # Add feedback to all matching areas
                for match_idx in matches:
                    if match_idx < len(key_areas):
                        area = str(key_areas[match_idx])
                        classified_feedback[area].append(feedback_texts[i])
                
                matched_count += 1
                
            logger.info(f"Successfully classified {matched_count} feedback items to key areas using embeddings")
            
        except Exception as e:
            logger.error(f"Error with embedding classification: {str(e)}")
            logger.info("Falling back to direct OpenAI classification")
            
            # Fall back to direct OpenAI classification using chunks
            # Process in chunks of 10 feedbacks at a time
            chunk_size = 10
            for i in range(0, len(feedback_items), chunk_size):
                logger.info(f"Processing feedback chunk {i//chunk_size + 1}/{(len(feedback_items) + chunk_size - 1)//chunk_size}")
                
                # Prepare the current chunk
                chunk = feedback_items[i:i + min(chunk_size, len(feedback_items) - i)]
                chunk_texts = [item["text"] for item in chunk]
                
                # Create prompt for classification
                areas_text = "\n".join([f"{j+1}. {area}" for j, area in enumerate(key_areas)])
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
                        model="gpt-4o-mini",
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
                                    first_area = str(key_areas[0])
                                    classified_feedback[first_area].append(feedback_text)
                                else:
                                    # Add to all matching categories
                                    for cat_idx in category_indices:
                                        if 0 <= cat_idx < len(key_areas):
                                            area = str(key_areas[cat_idx])
                                            classified_feedback[area].append(feedback_text)
                
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i//chunk_size + 1}: {str(chunk_error)}")
                    # Assign all feedbacks in this chunk to the first category as fallback
                    if key_areas:
                        first_area = str(key_areas[0])
                        classified_feedback[first_area].extend(chunk_texts)
                
                # Avoid rate limits
                await asyncio.sleep(0.5)
            
            logger.info("Completed fallback direct classification")
        
        # Log classification stats
        logger.info("Classification results:")
        for area, feedbacks in classified_feedback.items():
            logger.info(f"  - {area}: {len(feedbacks)} items")
        
        return {
            "key_areas": key_areas,
            "classified_feedback": classified_feedback
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}\n{traceback.format_exc()}")
        # Fall back to raw data processing
        logger.info("Falling back to raw data processing")
        return await analyze_raw_data_chunks(client, csv_data)

async def format_analysis_results(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the analysis results into the final structure.
    
    Args:
        analysis_results: Dictionary with key areas and classified feedback
        
    Returns:
        Formatted results matching the required output structure
    """
    key_areas = analysis_results.get("key_areas", [])
    classified_feedback = analysis_results.get("classified_feedback", {})
    
    # Format the results
    formatted_results = []
    
    # Process each key area
    for area in key_areas:
        # Ensure area is a string key
        area_str = str(area)
        area_feedbacks = classified_feedback.get(area_str, [])
        
        # For backward compatibility with old format
        if isinstance(area, dict):
            area_name = area.get("area", "Unknown Area")
            area_problem = area.get("problem", "") 
        else:
            area_name = area_str
            area_problem = f"Issues related to {area_str.lower()}"
        
        formatted_results.append({
            "key_area": area_name,
            "customer_problem": area_problem,
            "number_of_users": len(area_feedbacks),
            "raw_feedbacks": area_feedbacks
        })
    
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
async def analyze_feedback(file: UploadFile = File(...)) -> JSONResponse:
    """
    Analyzes customer feedback from an uploaded CSV file.
    
    Args:
        file: Uploaded file (expected to be CSV)
        
    Returns:
        JSONResponse with analysis results
    """
    client = create_client()
    job_id = f"feedback_analysis_{int(time.time())}"
    
    try:
        start_time = time.time()
        logger.info(f"[JOB {job_id}] Starting analysis of file: {file.filename}")
        
        # Read the uploaded file
        logger.info(f"[JOB {job_id}] Reading file content")
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"[JOB {job_id}] File size: {file_size/1024:.1f} KB")
        
        # Try to decode as text
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
            analysis_results = await process_csv_data(client, file_text)
        else:
            logger.info(f"[JOB {job_id}] File not detected as CSV, processing as raw text")
            analysis_results = await analyze_raw_data_chunks(client, file_text)
        
        # Format the results
        logger.info(f"[JOB {job_id}] Formatting final analysis results")
        formatted_results = await format_analysis_results(analysis_results)
        
        # Log summary statistics
        areas_count = len(formatted_results.get("analysis_results", []))
        total_items = formatted_results.get("summary", {}).get("total_feedback_items", 0)
        
        # Log the detected key areas and their counts
        logger.info(f"[JOB {job_id}] ANALYSIS SUMMARY: {areas_count} key areas, {total_items} total feedback items")
        for area in formatted_results.get("analysis_results", []):
            logger.info(f"[JOB {job_id}] - {area['key_area']}: {area['number_of_users']} feedback items")
        
        elapsed_time = time.time() - start_time
        logger.info(f"[JOB {job_id}] Analysis completed in {elapsed_time:.2f} seconds")
        timestamp = int(time.time())
        output_filename = f"feedback_analysis_{timestamp}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis results saved to {output_filename}")
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
async def standalone_analyze_feedback(file: UploadFile = File(...)) -> JSONResponse:
    """
    Standalone version of the analyze_feedback endpoint.
    
    Args:
        file: Uploaded file (expected to be CSV)
        
    Returns:
        JSONResponse with analysis results
    """
    return await analyze_feedback(file)

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