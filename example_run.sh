#!/bin/bash
set -e

python construction/construction.py --dataset crag --crag_line_id 2
# python qa/ToG/ToG.py --path "./question_2" --query "where did the ceo of salesforce previously work?"

# python construction/construction.py --dataset crag --crag_line_id 54
# python qa/ToG/ToG.py --path "./question_54" --query "what was mike epps's age at the time of next friday's release?"
#
# python construction/construction.py --dataset crag --crag_line_id 92
# python qa/ToG/ToG.py --path "./question_92" --query "what was the 76ers' record the year allen iverson won mvp?"
#
# python construction/construction.py --dataset crag --crag_line_id 98
# python qa/ToG/ToG.py --path "./question_98" --query "what age did ferdinand magelan discovered the philippines?"
#
# python construction/construction.py --dataset crag --crag_line_id 272
# python qa/ToG/ToG.py --path "./question_272" --query "what was taylor swifts age when she released her debut album?"
