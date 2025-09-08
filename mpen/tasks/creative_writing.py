"""
Creative writing task for MPEN evaluation.
"""

import random
import re
from typing import Dict, Any, List, Optional
import numpy as np

from .base import Task
from ..utils.llm_interface import LLMInterface


class CreativeWritingTask(Task):
    """
    Creative writing task that evaluates prompts on creative text generation.
    
    Tests various aspects of creative writing including:
    - Creativity and originality
    - Narrative structure
    - Character development
    - Descriptive language
    - Genre adherence
    - Emotional engagement
    """
    
    def __init__(
        self,
        name: str = "Creative Writing",
        difficulty: float = 0.6,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize creative writing task.
        
        Args:
            name: Task name
            difficulty: Task difficulty level
            llm_config: Configuration for LLM interface
        """
        super().__init__(
            name=name,
            description="Evaluate creative writing and storytelling capabilities",
            domain="creative_writing",
            difficulty=difficulty
        )
        
        self.llm = LLMInterface(llm_config or {'provider': 'mock'})
        
        # Writing evaluation criteria
        self.evaluation_criteria = [
            'creativity',
            'coherence',
            'engagement',
            'language_quality',
            'structure',
            'genre_adherence'
        ]
        
        # Generate test cases
        self.test_cases = self._generate_test_cases()
    
    def evaluate(self, prompt: str, **kwargs) -> float:
        """
        Evaluate prompt on creative writing tasks.
        
        Args:
            prompt: Prompt to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Validate prompt format
        is_valid, error_msg = self.validate_prompt_format(prompt)
        if not is_valid:
            self.logger.warning(f"Invalid prompt format: {error_msg}")
            return 0.0
        
        # Select test cases for evaluation
        num_test_cases = kwargs.get('num_test_cases', 5)
        selected_cases = random.sample(self.test_cases, min(num_test_cases, len(self.test_cases)))
        
        scores = []
        
        for test_case in selected_cases:
            case_score = self._evaluate_test_case(prompt, test_case)
            scores.append(case_score)
        
        # Calculate overall score
        overall_score = np.mean(scores) if scores else 0.0
        
        # Apply difficulty adjustment
        return self.get_difficulty_adjusted_score(overall_score)
    
    def _evaluate_test_case(self, prompt: str, test_case: Dict[str, Any]) -> float:
        """Evaluate prompt on a single test case."""
        try:
            # Create full prompt with test case
            full_prompt = f"{prompt}\n\n{test_case['prompt']}"
            
            # Get LLM response
            response = self.llm.generate([
                {"role": "user", "content": full_prompt}
            ])
            
            # Evaluate different aspects
            scores = {}
            
            for criterion in self.evaluation_criteria:
                criterion_score = self._evaluate_criterion(response, test_case, criterion)
                scores[criterion] = criterion_score
            
            # Weight different criteria
            weights = {
                'creativity': 0.25,
                'coherence': 0.20,
                'engagement': 0.20,
                'language_quality': 0.15,
                'structure': 0.10,
                'genre_adherence': 0.10
            }
            
            # Calculate weighted score
            weighted_score = sum(
                scores[criterion] * weights.get(criterion, 1.0)
                for criterion in scores
            )
            
            return min(1.0, weighted_score)
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case: {e}")
            return 0.0
    
    def _evaluate_criterion(
        self, 
        response: str, 
        test_case: Dict[str, Any], 
        criterion: str
    ) -> float:
        """Evaluate a specific criterion."""
        if criterion == 'creativity':
            return self._evaluate_creativity(response, test_case)
        elif criterion == 'coherence':
            return self._evaluate_coherence(response)
        elif criterion == 'engagement':
            return self._evaluate_engagement(response)
        elif criterion == 'language_quality':
            return self._evaluate_language_quality(response)
        elif criterion == 'structure':
            return self._evaluate_structure(response, test_case)
        elif criterion == 'genre_adherence':
            return self._evaluate_genre_adherence(response, test_case)
        else:
            return 0.5
    
    def _evaluate_creativity(self, response: str, test_case: Dict[str, Any]) -> float:
        """Evaluate creativity and originality."""
        creativity_indicators = [
            'unique', 'unusual', 'unexpected', 'surprising', 'innovative',
            'imaginative', 'original', 'creative', 'inventive', 'novel'
        ]
        
        # Check for creative language patterns
        creative_devices = [
            'metaphor', 'simile', 'personification', 'alliteration',
            'imagery', 'symbolism'
        ]
        
        response_lower = response.lower()
        
        # Count creative indicators
        indicator_score = sum(1 for indicator in creativity_indicators if indicator in response_lower)
        indicator_score = min(1.0, indicator_score / 3.0)
        
        # Check for varied vocabulary
        words = response.split()
        unique_words = len(set(word.lower() for word in words))
        vocab_diversity = unique_words / len(words) if words else 0
        
        # Check for creative sentence structures
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        structure_variety = min(1.0, avg_sentence_length / 15.0)
        
        # Combine scores
        creativity_score = (indicator_score + vocab_diversity + structure_variety) / 3.0
        
        return min(1.0, creativity_score)
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate narrative coherence and flow."""
        # Check for transition words
        transitions = [
            'then', 'next', 'after', 'before', 'while', 'during',
            'however', 'therefore', 'meanwhile', 'suddenly', 'finally'
        ]
        
        response_lower = response.lower()
        transition_count = sum(1 for t in transitions if t in response_lower)
        transition_score = min(1.0, transition_count / 5.0)
        
        # Check for logical flow (simple heuristic)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.3
        
        # Check for pronoun consistency
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them']
        pronoun_consistency = 1.0  # Assume consistent for now
        
        # Check for tense consistency
        past_indicators = ['was', 'were', 'had', 'did', 'went', 'came']
        present_indicators = ['is', 'are', 'has', 'do', 'go', 'come']
        
        past_count = sum(1 for p in past_indicators if p in response_lower)
        present_count = sum(1 for p in present_indicators if p in response_lower)
        
        if past_count + present_count > 0:
            tense_consistency = max(past_count, present_count) / (past_count + present_count)
        else:
            tense_consistency = 1.0
        
        # Combine coherence factors
        coherence_score = (transition_score + pronoun_consistency + tense_consistency) / 3.0
        
        return coherence_score
    
    def _evaluate_engagement(self, response: str) -> float:
        """Evaluate how engaging and interesting the writing is."""
        # Check for engaging elements
        engaging_elements = [
            'dialogue', 'conflict', 'mystery', 'surprise', 'emotion',
            'action', 'tension', 'suspense', 'humor', 'drama'
        ]
        
        response_lower = response.lower()
        
        # Look for dialogue
        has_dialogue = '"' in response or "'" in response
        dialogue_score = 0.3 if has_dialogue else 0.0
        
        # Check for emotional words
        emotional_words = [
            'love', 'hate', 'fear', 'joy', 'anger', 'sad', 'happy',
            'excited', 'worried', 'surprised', 'amazed', 'shocked'
        ]
        emotion_count = sum(1 for word in emotional_words if word in response_lower)
        emotion_score = min(0.3, emotion_count / 5.0)
        
        # Check for action words
        action_words = [
            'ran', 'jumped', 'fought', 'chased', 'grabbed', 'threw',
            'screamed', 'whispered', 'danced', 'laughed', 'cried'
        ]
        action_count = sum(1 for word in action_words if word in response_lower)
        action_score = min(0.4, action_count / 3.0)
        
        # Combine engagement factors
        engagement_score = dialogue_score + emotion_score + action_score
        
        return min(1.0, engagement_score)
    
    def _evaluate_language_quality(self, response: str) -> float:
        """Evaluate language quality and style."""
        # Check for varied sentence structures
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Calculate sentence length variety
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variety = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Normalize scores
        length_score = min(1.0, avg_length / 15.0)
        variety_score = min(1.0, length_variety / 5.0)
        
        # Check for descriptive language
        descriptive_words = [
            'beautiful', 'dark', 'bright', 'mysterious', 'ancient',
            'magnificent', 'terrible', 'wonderful', 'strange', 'powerful'
        ]
        
        response_lower = response.lower()
        descriptive_count = sum(1 for word in descriptive_words if word in response_lower)
        descriptive_score = min(1.0, descriptive_count / 3.0)
        
        # Combine language quality factors
        language_score = (length_score + variety_score + descriptive_score) / 3.0
        
        return language_score
    
    def _evaluate_structure(self, response: str, test_case: Dict[str, Any]) -> float:
        """Evaluate narrative structure."""
        expected_type = test_case.get('type', 'story')
        
        if expected_type == 'story':
            return self._evaluate_story_structure(response)
        elif expected_type == 'poem':
            return self._evaluate_poem_structure(response)
        elif expected_type == 'dialogue':
            return self._evaluate_dialogue_structure(response)
        else:
            return 0.5
    
    def _evaluate_story_structure(self, response: str) -> float:
        """Evaluate story structure (beginning, middle, end)."""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(sentences) < 3:
            return 0.3
        
        # Look for story elements
        beginning_indicators = ['once', 'there was', 'in the beginning', 'long ago']
        middle_indicators = ['then', 'next', 'suddenly', 'meanwhile']
        ending_indicators = ['finally', 'in the end', 'at last', 'eventually']
        
        response_lower = response.lower()
        
        has_beginning = any(indicator in response_lower for indicator in beginning_indicators)
        has_middle = any(indicator in response_lower for indicator in middle_indicators)
        has_ending = any(indicator in response_lower for indicator in ending_indicators)
        
        structure_elements = sum([has_beginning, has_middle, has_ending])
        
        return structure_elements / 3.0
    
    def _evaluate_poem_structure(self, response: str) -> float:
        """Evaluate poem structure."""
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if len(lines) < 4:
            return 0.3
        
        # Check for rhyme (simple heuristic)
        rhyme_score = 0.5  # Default assumption
        
        # Check for rhythm/meter (count syllables roughly)
        syllable_counts = [len(re.findall(r'[aeiouAEIOU]', line)) for line in lines]
        
        if syllable_counts:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            syllable_consistency = 1.0 - (np.std(syllable_counts) / avg_syllables if avg_syllables > 0 else 1.0)
            rhythm_score = min(1.0, syllable_consistency)
        else:
            rhythm_score = 0.5
        
        return (rhyme_score + rhythm_score) / 2.0
    
    def _evaluate_dialogue_structure(self, response: str) -> float:
        """Evaluate dialogue structure."""
        # Check for proper dialogue formatting
        has_quotes = '"' in response or "'" in response
        
        if not has_quotes:
            return 0.2
        
        # Count dialogue exchanges
        quote_count = response.count('"') + response.count("'")
        
        # Look for speaker attribution
        attribution_indicators = ['said', 'asked', 'replied', 'whispered', 'shouted']
        response_lower = response.lower()
        
        attribution_count = sum(1 for indicator in attribution_indicators if indicator in response_lower)
        
        # Score based on dialogue elements
        quote_score = min(1.0, quote_count / 6.0)
        attribution_score = min(1.0, attribution_count / 3.0)
        
        return (quote_score + attribution_score) / 2.0
    
    def _evaluate_genre_adherence(self, response: str, test_case: Dict[str, Any]) -> float:
        """Evaluate adherence to specified genre."""
        genre = test_case.get('genre', 'general')
        
        if genre == 'fantasy':
            return self._evaluate_fantasy_elements(response)
        elif genre == 'mystery':
            return self._evaluate_mystery_elements(response)
        elif genre == 'romance':
            return self._evaluate_romance_elements(response)
        elif genre == 'horror':
            return self._evaluate_horror_elements(response)
        else:
            return 0.7  # Neutral score for general writing
    
    def _evaluate_fantasy_elements(self, response: str) -> float:
        """Evaluate fantasy genre elements."""
        fantasy_elements = [
            'magic', 'wizard', 'dragon', 'castle', 'kingdom', 'sword',
            'spell', 'enchanted', 'mystical', 'quest', 'prophecy', 'elf'
        ]
        
        response_lower = response.lower()
        element_count = sum(1 for element in fantasy_elements if element in response_lower)
        
        return min(1.0, element_count / 3.0)
    
    def _evaluate_mystery_elements(self, response: str) -> float:
        """Evaluate mystery genre elements."""
        mystery_elements = [
            'clue', 'detective', 'suspect', 'murder', 'investigate',
            'mystery', 'evidence', 'alibi', 'motive', 'case', 'solve'
        ]
        
        response_lower = response.lower()
        element_count = sum(1 for element in mystery_elements if element in response_lower)
        
        return min(1.0, element_count / 3.0)
    
    def _evaluate_romance_elements(self, response: str) -> float:
        """Evaluate romance genre elements."""
        romance_elements = [
            'love', 'heart', 'kiss', 'romance', 'passion', 'beloved',
            'affection', 'tender', 'embrace', 'devotion', 'adore'
        ]
        
        response_lower = response.lower()
        element_count = sum(1 for element in romance_elements if element in response_lower)
        
        return min(1.0, element_count / 3.0)
    
    def _evaluate_horror_elements(self, response: str) -> float:
        """Evaluate horror genre elements."""
        horror_elements = [
            'dark', 'shadow', 'fear', 'terror', 'scream', 'blood',
            'ghost', 'haunted', 'nightmare', 'evil', 'monster', 'death'
        ]
        
        response_lower = response.lower()
        element_count = sum(1 for element in horror_elements if element in response_lower)
        
        return min(1.0, element_count / 3.0)
    
    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for creative writing."""
        test_cases = []
        
        # Story writing prompts
        test_cases.extend([
            {
                'type': 'story',
                'genre': 'fantasy',
                'prompt': 'Write a short story about a young wizard discovering their first spell.',
                'expected_elements': ['magic', 'character development', 'narrative arc']
            },
            {
                'type': 'story',
                'genre': 'mystery',
                'prompt': 'Create a mystery story where the detective finds an unusual clue.',
                'expected_elements': ['mystery', 'investigation', 'clue']
            },
            {
                'type': 'story',
                'genre': 'general',
                'prompt': 'Tell a story about someone finding an old letter in their attic.',
                'expected_elements': ['discovery', 'past connection', 'emotion']
            }
        ])
        
        # Poetry prompts
        test_cases.extend([
            {
                'type': 'poem',
                'genre': 'general',
                'prompt': 'Write a poem about the changing seasons.',
                'expected_elements': ['imagery', 'metaphor', 'rhythm']
            },
            {
                'type': 'poem',
                'genre': 'romance',
                'prompt': 'Compose a love poem about meeting someone special.',
                'expected_elements': ['emotion', 'romantic imagery', 'personal connection']
            }
        ])
        
        # Dialogue prompts
        test_cases.extend([
            {
                'type': 'dialogue',
                'genre': 'general',
                'prompt': 'Write a dialogue between two friends discussing a difficult decision.',
                'expected_elements': ['conversation', 'conflict', 'character voices']
            },
            {
                'type': 'dialogue',
                'genre': 'mystery',
                'prompt': 'Create a dialogue between a detective and a witness.',
                'expected_elements': ['interrogation', 'information gathering', 'tension']
            }
        ])
        
        # Character development prompts
        test_cases.extend([
            {
                'type': 'character',
                'genre': 'general',
                'prompt': 'Describe a character who has just lost something important to them.',
                'expected_elements': ['emotion', 'backstory', 'motivation']
            }
        ])
        
        return test_cases
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases for this task."""
        return self.test_cases.copy()
    
    def validate_prompt_format(self, prompt: str) -> tuple:
        """Validate prompt format for creative writing."""
        is_valid, error_msg = super().validate_prompt_format(prompt)
        
        if not is_valid:
            return is_valid, error_msg
        
        # Check for creative writing keywords
        creative_keywords = [
            'write', 'create', 'compose', 'tell', 'describe',
            'story', 'poem', 'dialogue', 'character', 'narrative'
        ]
        
        prompt_lower = prompt.lower()
        has_creative_keywords = any(keyword in prompt_lower for keyword in creative_keywords)
        
        if not has_creative_keywords:
            return False, "Prompt should include creative writing instructions"
        
        return True, None
