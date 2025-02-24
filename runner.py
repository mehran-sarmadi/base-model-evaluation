import logging
import argparse
import os
from typing import Dict, List, Tuple

# Persian tasks
from persian.deepsentipers import deepsentipers_main
from persian.farstail import farstail_main
from persian.pquad import pquad_main
from persian.parsinlu_mult import parsinlu_mult_main

# English tasks (uncomment functions when ready)
from english.squadv1 import squad_main
from english.sst2 import sst2_main
from english.mnli import mnli_main
from english.race import race_main

# Arabic tasks
from arabic.emotone import emotone_main
from arabic.arentail import arentail_main
from arabic.arcd import arcd_main

# Task configuration structure
LANGUAGE_TASKS: Dict[str, Dict[str, Tuple]] = {
    'fa': {
        'name': 'Persian',
        'tasks': [
            ('PQuAD', pquad_main),
            ('FarsTail', farstail_main),
            ('DeepSentiPers', deepsentipers_main),
            ('ParsiNLU_Multi', parsinlu_mult_main),
        ]
    },
    'en': {
        'name': 'English',
        'tasks': [
            ('CoLA', ),      
            ('SST-2	', ),      
            ('MRPC', ),      
            ('STS-B', ),
            ('QQP', ),
        ]
    },
    'ar': {
        'name': 'Arabic',
        'tasks': [
            ('Emotone', emotone_main),
            ('ArEntail', arentail_main),
            ('ARCD', arcd_main),
        ]
    }
}


def configure_logging(args: argparse.Namespace) -> logging.Logger:
    """Configure logging system with file and stream handlers."""
    log_dir = os.path.join(os.getcwd(), "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    model_safe_name = args.model_name_or_path.replace('/', '_').strip('_')
    log_filename = f"{model_safe_name}_{args.language}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_tasks(language: str, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Execute tasks for specified language."""
    if language not in LANGUAGE_TASKS:
        logger.warning(f"No tasks configured for language: {language}")
        return

    lang_config = LANGUAGE_TASKS[language]
    logger.info(f"\n\n{'=' * 35} Testing on {lang_config['name']} Datasets {'=' * 35}\n")

    for task_name, task_function in lang_config['tasks']:
        logger.info(f"\n{'-' * 20} {task_name} {'-' * 20}\n")
        task_function(
            model_name_or_path=args.model_name_or_path,
            logger=logger,
            output_dir=args.output_dir,
            tokenizer_name_or_path=args.tokenizer_name_or_path or args.model_name_or_path
        )


def main(args: argparse.Namespace) -> None:
    """Main execution flow."""
    logger = configure_logging(args)
    logger.info("\n\n%s", "=" * 50)
    logger.info("%20s STARTING EXPERIMENTS %20s", "", "")
    logger.info("%s\n\n", "=" * 50)

    target_languages = ['fa', 'en', 'ar'] if args.language == 'all' else [args.language]
    
    for lang in target_languages:
        run_tasks(lang, args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual NLP Evaluation Framework")
    
    parser.add_argument('--model_name_or_path', 
        type=str, 
        required=True,
        help="HuggingFace model identifier or local path")
    
    parser.add_argument('--tokenizer_name_or_path', 
        type=str, 
        default=None,
        help="Tokenizer name/path (defaults to model name if not specified)")
    
    parser.add_argument('--language', 
        type=str, 
        required=True, 
        choices=['all', 'en', 'ar', 'fa'],
        help="Language(s) to evaluate on")
    
    parser.add_argument('--output_dir', 
        type=str, 
        default='./results',
        help="Output directory for evaluation results")

    args = parser.parse_args()
    main(args)