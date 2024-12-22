class ConversationalCodeEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.code_evaluator = CodeEvaluationMetrics(tokenizer)
        
    def extract_code_and_context(self, text):
        """
        Extract both code blocks and their surrounding explanation
        """
        # Split text into segments (code and non-code)
        segments = []
        code_blocks = []
        
        # Pattern for code blocks
        pattern = r'```(?:python)?(.*?)```'
        
        # Split the text and preserve both code and context
        parts = re.split(pattern, text, flags=re.DOTALL)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Non-code context
                segments.append(('context', part.strip()))
            else:  # Code block
                segments.append(('code', part.strip()))
                code_blocks.append(part.strip())
                
        return segments, code_blocks

    def evaluate_response(self, reference, generated):
        """
        Evaluate both the conversational aspects and code quality
        """
        metrics = {
            'conversation': {},
            'code': {},
            'overall': {}
        }
        
        # Extract code and context from both reference and generated text
        ref_segments, ref_code = self.extract_code_and_context(reference)
        gen_segments, gen_code = self.extract_code_and_context(generated)
        
        # Evaluate conversation flow and completeness
        metrics['conversation'] = self.evaluate_conversation(
            ref_segments, 
            gen_segments
        )
        
        # Evaluate code if present
        if ref_code and gen_code:
            for ref, gen in zip(ref_code, gen_code):
                code_metrics = self.code_evaluator.evaluate_generation(ref, gen)
                for key, value in code_metrics.items():
                    metrics['code'][key] = metrics['code'].get(key, []) + [value]
        
        # Calculate overall metrics
        metrics['overall'] = self.calculate_overall_score(metrics)
        
        return metrics

    def evaluate_conversation(self, ref_segments, gen_segments):
        """
        Evaluate the conversational aspects of the response
        """
        metrics = {
            'explanation_completeness': 0.0,
            'context_preservation': 0.0,
            'format_consistency': 0.0
        }
        
        # Check if explanation surrounds code blocks
        has_intro = any(seg[0] == 'context' for seg in gen_segments[:1])
        has_conclusion = any(seg[0] == 'context' for seg in gen_segments[-1:])
        metrics['explanation_completeness'] = (has_intro + has_conclusion) / 2
        
        # Check context preservation
        ref_contexts = [seg[1] for seg in ref_segments if seg[0] == 'context']
        gen_contexts = [seg[1] for seg in gen_segments if seg[0] == 'context']
        metrics['context_preservation'] = self.calculate_context_similarity(
            ref_contexts, 
            gen_contexts
        )
        
        # Check format consistency
        metrics['format_consistency'] = self.check_format_consistency(gen_segments)
        
        return metrics

    def calculate_context_similarity(self, ref_contexts, gen_contexts):
        """
        Calculate similarity between reference and generated explanations
        """
        if not ref_contexts or not gen_contexts:
            return 0.0
            
        # Use BLEU or other similarity metrics
        return self.code_evaluator.calculate_bleu_score(
            ' '.join(ref_contexts),
            ' '.join(gen_contexts)
        )

    def check_format_consistency(self, segments):
        """
        Check if the response maintains proper formatting
        """
        # Check for alternating context-code pattern
        is_alternating = all(
            segments[i][0] != segments[i+1][0] 
            for i in range(len(segments)-1)
        )
        
        # Check for proper code block formatting
        has_code_blocks = any(seg[0] == 'code' for seg in segments)
        
        return (is_alternating and has_code_blocks)

    def calculate_overall_score(self, metrics):
        """
        Calculate an overall quality score
        """
        scores = {
            'conversation_score': sum(metrics['conversation'].values()) / len(metrics['conversation']),
            'code_score': np.mean([
                np.mean(values) for values in metrics['code'].values()
            ]) if metrics['code'] else 0.0
        }
        
        # Weighted average of conversation and code scores
        scores['overall_score'] = 0.4 * scores['conversation_score'] + 0.6 * scores['code_score']
        
        return scores