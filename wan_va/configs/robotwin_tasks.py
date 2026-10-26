"""RoboTwin task splits shared by training configs and evaluation."""

ROBOTWIN_EVAL_TASKS = (
    'place_object_scale', 'stamp_seal', 'open_microwave',
    'move_stapler_pad', 'place_bread_basket', 'place_empty_cup',
    'stack_blocks_three',
)

ROBOTWIN_SUBSET12_TASKS = (
    'move_pillbottle_pad', 'place_mouse_pad', 'place_container_plate',
    'place_bread_skillet', 'place_can_basket', 'pick_dual_bottles',
    'open_laptop', 'put_object_cabinet', 'beat_block_hammer',
    'stack_blocks_two', 'stack_bowls_two', 'stack_bowls_three',
)

# Independent one-GPU servers; long tasks get a less crowded worker.
ROBOTWIN_EVAL_ASSIGNMENTS = (
    ('stack_blocks_three',),
    ('open_microwave', 'place_empty_cup'),
    ('stamp_seal', 'move_stapler_pad'),
    ('place_bread_basket', 'place_object_scale'),
)
