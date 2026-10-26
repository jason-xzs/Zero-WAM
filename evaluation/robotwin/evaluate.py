"""Evaluate seven unique tasks using four independent single-GPU servers."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import signal
from statistics import mean, pstdev
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener

from wan_va.configs.robotwin_tasks import (
    ROBOTWIN_EVAL_ASSIGNMENTS as EVAL_ASSIGNMENTS,
    ROBOTWIN_EVAL_TASKS as EVAL_TASKS,
)

ROOT = Path(__file__).resolve().parents[2]


def make_plan(args):
    plan = []
    for slot, (gpu, tasks) in enumerate(zip(args.gpus, EVAL_ASSIGNMENTS)):
        port = args.start_port + slot
        server = [args.server_python, '-m', 'torch.distributed.run',
                  '--nproc_per_node=1', '--master_port', str(args.master_port + slot),
                  '-m', 'wan_va.wan_va_server', '--config-name', 'robotwin',
                  '--port', str(port), '--save_root',
                  str(args.save_root / 'server_visualization' / f'gpu_{slot}')]
        clients = []
        for seed in args.seeds:
            for task in tasks:
                command = [args.client_python, '-m', 'evaluation.robotwin.eval_policy_client_openpi',
                           '--config', str(args.robotwin_root / 'policy/ACT/deploy_policy.yml'),
                           '--port', str(port), '--save_root', str(args.save_root / f'seed_{seed}'),
                           '--video_guidance_scale', '-1', '--action_guidance_scale', '1',
                           '--icl_guidance_scale', '5', '--icl_human_video_map',
                           str(ROOT / 'evaluation/robotwin/robotwin_icl_human_videos.py'),
                           '--icl_latent_root', str(args.icl_latent_root),
                           '--icl_seed', str(seed), '--test_num', str(args.test_num),
                           '--overrides', '--task_name', task, '--task_config', 'demo_clean',
                           '--train_config_name', '0', '--model_name', '0', '--ckpt_setting', '0',
                           '--seed', str(seed), '--policy_name', 'ACT']
                clients.append({'seed': seed, 'task': task, 'command': command})
        plan.append({'slot': slot, 'gpu': gpu, 'port': port, 'tasks': list(tasks),
                     'server_command': server, 'clients': clients})
    return plan


def summarize(args):
    rates = {task: [] for task in EVAL_TASKS}
    for seed in args.seeds:
        for task in EVAL_TASKS:
            path = (args.save_root / f'seed_{seed}' / f'stseed-{100000 * (1 + seed)}'
                    / 'metrics' / task / 'res.json')
            result = json.loads(path.read_text())
            if result['total_num'] != args.test_num:
                raise RuntimeError(f'Incomplete evaluation: {path}')
            rates[task].append(result['succ_rate'])
    per_seed_macro = [mean(rates[task][i] for task in EVAL_TASKS) for i in range(len(args.seeds))]
    result = {
        'seeds': args.seeds, 'rollouts_per_task_per_seed': args.test_num,
        'per_task': {task: {'rates': values, 'mean': mean(values), 'std_ddof0': pstdev(values)}
                     for task, values in rates.items()},
        'macro_per_seed': per_seed_macro, 'macro_mean': mean(per_seed_macro),
        'macro_std_ddof0': pstdev(per_seed_macro), 'rate_unit': 'fraction_0_to_1',
    }
    (args.save_root / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs=4, default=['0', '1', '2', '3'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--start-port', type=int, default=29556)
    parser.add_argument('--master-port', type=int, default=29661)
    parser.add_argument('--server-start-timeout', type=int, default=900)
    parser.add_argument('--server-python', default=sys.executable)
    parser.add_argument('--client-python', default=sys.executable)
    parser.add_argument('--model-path', default=os.environ.get('MODEL_PATH'))
    parser.add_argument('--robotwin-root', type=Path, default=os.environ.get('ROBOTWIN_ROOT'))
    parser.add_argument('--icl-latent-root', type=Path, default=os.environ.get(
        'ICL_LATENT_ROOT', str(ROOT / 'data/HumanGen/human_latents/robotwin')))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != 4 or len(set(args.seeds)) != len(args.seeds) or args.test_num < 1:
        parser.error('Use four distinct GPUs, distinct seeds, and a positive test count')
    if args.dry_run and args.robotwin_root is None:
        args.robotwin_root = Path('/ROBOTWIN_ROOT')
    if not args.summarize_only and (args.robotwin_root is None or (not args.dry_run and not args.model_path)):
        parser.error('Set MODEL_PATH and ROBOTWIN_ROOT, or pass their corresponding options')
    args.save_root = args.save_root.expanduser().resolve()
    args.icl_latent_root = args.icl_latent_root.expanduser().resolve()
    if args.summarize_only:
        summarize(args)
        return
    args.robotwin_root = args.robotwin_root.expanduser().resolve()
    server_ports = set(range(args.start_port, args.start_port + 4))
    distributed_ports = set(range(args.master_port, args.master_port + 4))
    if server_ports & distributed_ports or not all(1024 <= p <= 65535 for p in server_ports | distributed_ports):
        parser.error('Server and distributed port ranges must be valid and disjoint')
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if (args.save_root / 'run-plan.json').exists():
        parser.error('Use a new --save-root for each evaluation to preserve previous results')
    # The inference server needs the complete inference model root, not a bare
    # transformer-only training checkpoint. Asset preparation is external.
    for component in ('transformer', 'vae', 'tokenizer', 'text_encoder'):
        if not (Path(args.model_path).expanduser() / component).is_dir():
            parser.error(f'Inference MODEL_PATH is missing component: {component}')
    logs = args.save_root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    (args.save_root / 'run-plan.json').write_text(json.dumps(plan, indent=2))
    children = {}
    lock = threading.Lock()
    stopping = threading.Event()
    servers = []
    executor = None

    def start(command, env, cwd, logfile):
        with lock:
            if stopping.is_set():
                raise RuntimeError('Evaluation stopping')
            with logfile.open('w') as output:
                process = subprocess.Popen(command, env=env, cwd=cwd, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            children[process.pid] = process
            return process

    def environment(gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, MODEL_PATH=str(Path(args.model_path).expanduser().resolve()),
                   ICL_CFG='5', TARGET_TEXT_CFG='-1', ZERO_WAM_CPU_OFFLOAD='0',
                   PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(args.robotwin_root), env.get('PYTHONPATH', '')])
        env['LD_LIBRARY_PATH'] = '/usr/lib64:/usr/lib:' + env.get('LD_LIBRARY_PATH', '')
        return env

    def check_servers():
        for process in servers:
            if process.poll() is not None:
                raise RuntimeError(f'Inference server {process.pid} exited: {process.returncode}; see {logs}')

    def worker(item):
        env = environment(item['gpu'])
        for client in item['clients']:
            logfile = logs / f"client_gpu{item['slot']}_seed{client['seed']}_{client['task']}.log"
            process = start(client['command'], env, args.robotwin_root, logfile)
            returncode = process.wait()
            with lock:
                children.pop(process.pid, None)
            if returncode:
                raise RuntimeError(f'Client failed ({returncode}): {logfile}')
            print(f"Completed seed={client['seed']} task={client['task']} gpu={item['gpu']}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for item in plan:
            env = environment(item['gpu'])
            env['TORCHINDUCTOR_CACHE_DIR'] = str(args.save_root / 'compile_cache' / f"gpu_{item['slot']}")
            servers.append(start(item['server_command'], env, ROOT, logs / f"server_gpu{item['slot']}.log"))
        deadline = time.monotonic() + args.server_start_timeout
        opener = build_opener(ProxyHandler({}))
        while True:
            check_servers()
            ready = True
            for item in plan:
                try:
                    with opener.open(f"http://127.0.0.1:{item['port']}/healthz", timeout=1) as response:
                        ready = ready and response.status == 200
                except OSError:
                    ready = False
            if ready:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Inference servers not ready; see {logs}')
            time.sleep(2)
        print('All four inference servers are ready', flush=True)
        executor = ThreadPoolExecutor(max_workers=4)
        pending = {executor.submit(worker, item) for item in plan}
        while pending:
            completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            check_servers()
            for future in completed:
                future.result()
        summarize(args)
    finally:
        with lock:
            stopping.set()
            active = list(children.values())
        for process in active:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for process in active:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
