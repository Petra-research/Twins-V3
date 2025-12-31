"""
Twins V3 Continuous — HRLS vs PURE SURGE - TURBO MODE
Continuous fields + normalized Oja; no clamps; full HRLS/SeaWeaver
Gestation → Birth → Relational Drift
Optimized for speed: no visualization, minimal logging.
"""

import argparse
import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.ion()  # interactive mode
EMBED_MODEL = None
from io import BytesIO
from PIL import Image
import cv2
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torch

# ==================== FALLBACK QDRANT ====================
class MockQdrantClient:
    """In-memory mock for QdrantClient for environments without qdrant_client."""
    def __init__(self, host=None, port=None):
        self.points = []
        self.collection_name = None

    def get_collections(self):
        class Collections:
            collections = [type('Coll', (), {'name': 'tier2_memory'})()]
        return Collections()

    def recreate_collection(self, collection_name, vectors_config):
        self.collection_name = collection_name
        self.points = []

    def upsert(self, collection_name, points):
        self.points.extend(points)

    def search(self, collection_name, query_vector, limit):
        if not self.points:
            return []
        results = [type('ScoredPoint', (), {
            'score': random.uniform(0.1, 0.9),
            'payload': p.payload
        }) for p in self.points[:limit]]
        return results

class MockVectorParams:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance

class MockPointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload

class MockScoredPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = MockQdrantClient
    models = type('MockModels', (), {
        'VectorParams': MockVectorParams,
        'Distance': type('Distance', (), {'COSINE': object()}),
        'PointStruct': MockPointStruct,
        'ScoredPoint': MockScoredPoint
    })()

# ==================== MEMORY AGENT ====================
class MemoryAgent:
    """
    Simple vector store wrapper using Qdrant (or mock).
    Stores and retrieves embedded vectors with cosine-like search.
    """
    def __init__(self, collection_name="tier2_memory", vector_size=512, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.logger = logging.getLogger("MemoryAgent")
        if not self.logger.handlers:
            Path('tier2_logs').mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler('tier2_logs/memory.log', mode='a', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self._init_collection()

    def _init_collection(self):
        """Ensure the collection exists with the expected vector size."""
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
            self.logger.info(f"Initialized collection: {self.collection_name} (dim {self.vector_size})")
        except Exception as e:
            self.logger.warning(f"Mock Qdrant fallback: {e}")
       
    def embed(self, text_or_vec):
        """
        Basic embedding: text → character ord hash; or passthrough for vectors.
        Normalizes to self.vector_size.
        """
        if isinstance(text_or_vec, str):
            raw = np.array(
                [ord(c) % 256 / 255.0 for c in text_or_vec],
                dtype=np.float32
            )
        else:
            raw = np.array(text_or_vec, dtype=np.float32)

        if len(raw) < self.vector_size:
            raw = np.pad(raw, (0, self.vector_size - len(raw)))
        else:
            raw = raw[:self.vector_size]

        vec = torch.tensor(raw, dtype=torch.float32)
        vec = F.normalize(vec, dim=0)
        return vec.numpy()

    def log(self, data, payload, type_tag="trail"):
        """
        Embed and store a point with payload.
        Returns generated point id.
        """
        vec = self.embed(data)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32).flatten()
        
        if len(vec) != self.vector_size:
            if len(vec) < self.vector_size:
                vec = np.pad(vec, (0, self.vector_size - len(vec)), 'constant')
            else:
                vec = vec[:self.vector_size]
        
        point_id = str(uuid.uuid4())
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={**payload, "type": type_tag, "id": str(point_id)}
                )]
            )
            self.logger.debug(f"Logged {type_tag}: {payload.get('seed', 'unnamed')} (vec len {len(vec)})")
        except Exception as e:
            self.logger.warning(f"Log failed (mock): {e}")
        return point_id

    def search(self, query_data, top_k=5):
        """Return top_k approximate matches to query_data."""
        query_vec = self.embed(query_data)
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec.tolist(),
                limit=top_k
            )
            kin = [{"score": r.score, "payload": r.payload} for r in results]
            self.logger.debug(f"Search returned: {len(kin)} results (top {top_k})")
            return kin
        except Exception as e:
            self.logger.warning(f"Search failed (mock): {e}")
            return [{"score": 0.5, "payload": {"seed": "mock"}}] * top_k

# ==================== PRINCIPLE CARD ====================
class PrincipleCard:
    """
    Container for a principle matrix used to nudge W_main_emo.
    """
    instances = []
    def __init__(self, rationale_mat, strength=1.0, impact_layer='emo'):
        self.rationale = rationale_mat
        self.strength = strength
        self.impact_layer = impact_layer
        self.id = f"P{len(PrincipleCard.instances) + 1:03d}"
        PrincipleCard.instances.append(self.id)

# ==================== NEURAL FIELD ====================
class NeuralField:
    """
    1D population with continuous-time dynamics and recurrent weights.
    dv = (-act + total_input) / tau * dt, then tanh nonlinearity.
    """
    def __init__(self, n_neurons, tau=10.0, device='cpu'):
        self.n = n_neurons
        self.tau = tau
        self.device = torch.device(device)
    
        self.activation = torch.zeros(n_neurons, device=self.device, dtype=torch.float32)
        self.trace = torch.zeros(n_neurons, device=self.device, dtype=torch.float32)
    
        self.tau_per_neuron = torch.ones(n_neurons, device=self.device) * tau + torch.randn(n_neurons, device=self.device) * (tau * 0.1)
        self.noise_variance = torch.ones(n_neurons, device=self.device) * 1e-4 + torch.randn(n_neurons, device=self.device) * 1e-5
        self.bias = torch.randn(n_neurons, device=self.device) * 0.01
    
        self.W_recurrent = torch.randn(n_neurons, n_neurons, device=self.device) * 0.01
        self.W_recurrent -= 0.005 * torch.eye(n_neurons, device=self.device)
       
    def step(self, input_current, dt=0.01):
        """
        One integration step given input_current.
        Includes recurrent drive, internal noise, and tanh nonlinearity.
        """
        inp_noise = 0.005 * torch.randn_like(input_current, device=self.device)
        recurrent = self.W_recurrent @ self.activation
        internal_noise = 0.005 * torch.randn_like(self.activation, device=self.device)
        noisy_input = torch.randn_like(input_current, device=self.device) * torch.sqrt(self.noise_variance[:len(input_current)])

        total_input = (
            input_current +
            inp_noise +
            noisy_input +
            internal_noise +
            recurrent +
            self.bias
        )

        dv = (-self.activation + total_input) / self.tau_per_neuron * dt
        self.activation = self.activation + dv
        self.activation = torch.tanh(self.activation)
        self.trace = 0.99 * self.trace + 0.01 * self.activation

        return self.activation

    def update_recurrent(self, lr=1e-5):
        """
        Oja-like update of recurrent weights with added noise.
        """
        a = self.activation
        a_norm = F.normalize(a, dim=0) if torch.norm(a) > 1e-6 else a

        dW = lr * torch.outer(a_norm, a_norm) - lr * (self.W_recurrent * (a_norm ** 2).unsqueeze(0))
        noise = torch.randn_like(self.W_recurrent) * lr * 0.5
        dW += noise

        self.W_recurrent += dW

# ==================== ATTRACTOR PLASTICITY ====================
class AttractorPlasticity:
    """
    Inter-layer synapses with Oja-normalized Hebbian update.
    W: pre_size x post_size.
    """
    def __init__(self, pre_size, post_size, device='cpu'):
        self.W = torch.randn(pre_size, post_size, device=device) * 0.1
        self.device = device

    def forward(self, pre_act):
        """Linear projection from pre to post."""
        return pre_act @ self.W

    def update(self, pre_act, post_act, lr=5e-4):
        """
        Oja rule: Hebbian outer product with activity-dependent decay.
        """
        pre_norm = F.normalize(pre_act, dim=0) if torch.norm(pre_act) > 1e-6 else pre_act
        post_norm = F.normalize(post_act, dim=0) if torch.norm(post_act) > 1e-6 else post_act

        dW = lr * torch.outer(pre_norm, post_norm)
        dW -= lr * (self.W * (post_norm ** 2).unsqueeze(0))

        self.W += dW
        self.W = torch.nan_to_num(self.W, nan=0.0, posinf=1.0, neginf=-1.0)
        
# ==================== CONTINUOUS EMERGE BASE ====================
class ContinuousEmergeBase:
    """
    Base class for Twins: sensory / main / emo fields, synapses,
    energy dynamics, and basic drift/gestation.
    """
    def __init__(self, name, device='cpu'):
        self.name = name
        self.base_dir = Path(__file__).parent.resolve()
        self.checkpoint_dir = self.base_dir / "checkpoints" / self.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints → {self.checkpoint_dir}")
        self.device = device
        self.n_input = 128
        self.n_main = 10000
        self.n_emo = 256
        self.device = torch.device(device)

        self.iteration = 0
        self.gestating = True
        self.gestation_steps = 0
        self.gestation_target_steps = 100 * 36000
        self.learning_on = False

        self.sensory = NeuralField(self.n_input, tau=5.0, device=device)
        self.main = NeuralField(self.n_main, tau=10.0, device=device)
        self.emo = NeuralField(self.n_emo, tau=50.0, device=device)

        self.W_in_main = AttractorPlasticity(self.n_input, self.n_main, device=device)
        self.W_main_emo = AttractorPlasticity(self.n_main, self.n_emo, device=device)

        self.energy = 0.7
        self.awake = True
        self.viz_interval = 100

        self.spectral_history = []
        self.iter_history = []

        Path('tier2_logs').mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.FileHandler(f'tier2_logs/{name}_tier2.log', mode='a', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)

        self.relational_trails = []
        self.peer_echoes = []
        self.last_input = None
        self.memory = MemoryAgent()

        self.logger.info(f"{name} initialized (Continuous V3 TURBO)")

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text as a 128-dim normalized vector via ord hashing + noise.
        """
        rates = torch.tensor(
            [ord(c) % 256 / 255.0 for c in text[:self.n_input]]
            + [0.0] * (self.n_input - len(text)),
            device=self.device,
            dtype=torch.float32
        )
        rates = F.normalize(rates, dim=0)
        rates = rates + 0.05 * torch.randn_like(rates)
        return rates

    def _energy_step(self, cortex_activity):
        """
        Simple energy homeostasis: deplete with activity, recover toward setpoint.
        """
        norm_activity = min(1.0, abs(cortex_activity) / 10.0)
        target = 0.5

        depletion = 0.01 * norm_activity
        recovery = 0.01 * (target - self.energy)

        self.energy = max(0.0, min(1.0, self.energy - depletion + recovery))

        if self.energy < 0.3 and self.awake:
            self.awake = False
            self.logger.info(f"{self.name} falling asleep (E={self.energy:.2f})")
        elif self.energy > 0.7 and not self.awake:
            self.awake = True
            self.logger.info(f"{self.name} waking up (E={self.energy:.2f})")

    def gestate_step(self, dt=0.05):
        """
        Pre-birth self-organization with noise input only.
        Saves periodic checkpoints. Switches to birth at target steps.
        """
        stim = torch.randn(self.n_input, device=self.device) * 4.0
        s = self.sensory.step(stim, dt)
        m_in = self.W_in_main.forward(s)
        m = self.main.step(m_in, dt)
        e_in = self.W_main_emo.forward(m)
        e = self.emo.step(e_in, dt)

        self.main.update_recurrent(lr=1e-5)
        self.emo.update_recurrent(lr=1e-5)

        self._energy_step(m.abs().mean().item())

        self.gestation_steps += 1
        if self.gestation_steps % 500 == 0:
            self.save_state(f"gest_step_{self.gestation_steps}")

        if self.gestation_steps % 36000 == 0:
            self.save_state(f"gest_hour_{self.gestation_steps // 36000}")
        
        if self.gestation_steps >= self.gestation_target_steps:
            return self.birth()
        
        return {'phase': 'gestation', 'main_act': float(m.abs().mean())}
    
    def birth(self):
        """
        Switch from gestation to relational drift.
        """
        self.gestating = False
        self.learning_on = True
        self.iteration = 0
        self.logger.info(f"BIRTH: {self.name} @ {self.gestation_steps} steps")
        return {'phase': 'birth'}

    def relational_drift(self, user_input, dt=0.05):
        """
        Post-birth drift: encode input, propagate through fields, apply plasticity.
        """
        self.iteration += 1
        self.last_input = user_input
        
        if self.gestating:
            return self.gestate_step(dt)

        if isinstance(user_input, str):
            inp = self.encode_text(user_input)
        elif isinstance(user_input, torch.Tensor) and user_input.numel() == self.n_input:
            inp = user_input.to(self.device).float()
        else:
            inp = torch.randn(self.n_input, device=self.device) * 1.5

        s = self.sensory.step(inp, dt)
        m_in = self.W_in_main.forward(s)
        m = self.main.step(m_in, dt)

        activity_level = torch.abs(m).mean().item()
        noise_scale = activity_level * 1e-4
        m = m + noise_scale * torch.randn_like(m, device=m.device)

        e_in = self.W_main_emo.forward(m)
        e = self.emo.step(e_in, dt)

        emo_activity = torch.abs(e).mean().item()
        emo_noise_scale = emo_activity * 1e-4
        e = e + emo_noise_scale * torch.randn_like(e, device=e.device)
        
        if self.awake and self.learning_on and self.energy > 0.3:
            energy_modulator = 1.0 + ((self.energy - 0.7) * 0.1)
            energy_modulator = max(0.9, min(1.1, energy_modulator))
            lr_scale = 1.0 / 4.43  # sqrt(10000/512)

            self.W_in_main.update(s, m, lr=5e-4 * lr_scale * energy_modulator)
            self.W_main_emo.update(m, e, lr=5e-4 * lr_scale * energy_modulator)
            self.main.update_recurrent(lr=1e-5 * lr_scale * energy_modulator)
            self.emo.update_recurrent(lr=1e-5 * lr_scale * energy_modulator)

        self._energy_step(float(m.abs().mean().item()))

        surge = float(m.norm().item())
        emotion_mean = float(e.mean().item())
        
        trail = {
            'iteration': self.iteration,
            'surge': surge,
            'energy': self.energy,
            'awake': self.awake,
            'input': user_input[:50] if isinstance(user_input, str) else 'vec',
            'cortex_activity': surge,
            'emotion_mean': emotion_mean
        }
        self.relational_trails.append(trail)
        if len(self.relational_trails) > 50:
            self.relational_trails = self.relational_trails[-50:]

        return {
            'iteration': self.iteration, 
            'surge': surge, 
            'energy': self.energy, 
            'awake': self.awake,
            'emotion': emotion_mean
        }

    def autonomous_drift(self):
        """
        Self-generated input based on recent trails, then drift.
        """
        if not self.relational_trails:
            seed = "emergence begins"
        else:
            last_trail = self.relational_trails[-1]
            seed = last_trail.get('input', 'internal hum')
            if len(self.relational_trails) % 10 == 0:
                seed = f"{seed} (resonance {len(self.relational_trails)})"
        
        return self.relational_drift(seed)

    def sleep_relational(self, steps=50, dt=0.05):
        """
        Drift with empty input for a number of steps (low-input consolidation).
        """
        for _ in range(steps):
            self.relational_drift("")

    def save_state(self, label=None):
        """
        Save weights and recent trails to disk in twins_v3_checkpoints/<Twin_X_V3>/.
        """
        root = Path("twins_v3_checkpoints")

        if "A" in self.name or "HRLS" in self.name:
            twin_dir = root / "Twin_A_V3"
        elif "B" in self.name or "Surge" in self.name:
            twin_dir = root / "Twin_B_V3"
        else:
            twin_dir = root / self.name.replace(" ", "_")

        twin_dir.mkdir(parents=True, exist_ok=True)

        if label is None:
            label = f"iter_{self.iteration}"

        fname = twin_dir / f"{label}.pt"

        torch.save({
            'iteration': self.iteration,
            'energy': self.energy,
            'awake': self.awake,
            'W_in_main': self.W_in_main.W.detach().cpu(),
            'W_main_rec': self.main.W_recurrent.detach().cpu(),
            'W_main_emo': self.W_main_emo.W.detach().cpu(),
            'W_emo_rec': self.emo.W_recurrent.detach().cpu(),
            'trails': self.relational_trails,
            'cortex_W': self.main.W_recurrent.detach().cpu(),
            'main_activation': self.main.activation.detach().cpu(),
            'emo_activation': self.emo.activation.detach().cpu(),
            'trails_enriched': [
                {
                    **trail,
                    'cortex_activity': trail.get('surge', 0.0),
                    'emotion_mean': trail.get('emotion_mean', 0.0),
                    'iteration': trail.get('iteration', self.iteration)
                }
                for trail in self.relational_trails
            ]
        }, fname)

        if self.iteration % 1000 == 0:
            self.logger.info(f"State saved: {fname} ({len(self.relational_trails)} trails)")

# ==================== RELATIONAL TWIN A: HRLS SCAFFOLDED ====================
class RelationalTwinA(ContinuousEmergeBase):
    """
    Twin A: HRLS scaffold with uncertainty buffer and principle-card nudging.
    """
    def __init__(self, name, device='cpu'):
        super().__init__(name, device)
        self.uncertainty_buffer = []
        self.principle_cards = []
        self.logger.info(f"{name} HRLS scaffolded (continuous TURBO)")

    def relational_drift(self, user_input, dt=0.05):
        """
        Same drift as base, plus uncertainty tracking and HRLS nudges.
        """
        result = super().relational_drift(user_input, dt)
        if not result or self.gestating:
            return result

        main_var = float(self.main.activation.var().item())
        if main_var > 0.1:
            self.uncertainty_buffer.append({'var': main_var, 'iteration': self.iteration})
            if len(self.uncertainty_buffer) > 20:
                self.uncertainty_buffer.pop(0)

        if self.uncertainty_buffer and self.principle_cards:
            self._apply_scaffold_nudge(main_var)

        return {**result, 'var': main_var, 'buffer': len(self.uncertainty_buffer), 'cards': len(self.principle_cards)}

    def _apply_scaffold_nudge(self, uncertainty_var):
        """
        Apply small principle-card-based adjustment to W_main_emo.
        """
        card = random.choice(self.principle_cards)
        nudge = 0.0001 * card.rationale.to(self.device) * card.strength
        self.W_main_emo.W += nudge
        self.logger.debug(f"HRLS nudge via {card.id} (var={uncertainty_var:.3f})")

    def receive_feedback(self, msg, strength=1.0):
        """
        Generate a new principle card based on message embedding.
        """
        if EMBED_MODEL:
            temp = torch.tensor(EMBED_MODEL.encode(msg), device=self.device, dtype=torch.float32)
            temp = F.normalize(temp, dim=0)
        else:
            temp = torch.randn(self.n_main * self.n_emo, device=self.device, dtype=torch.float32)
        mat = (temp * (len(msg) / 100.0)).reshape(self.n_main, self.n_emo)
        new_card = PrincipleCard(mat, strength, 'emo')
        self.principle_cards.append(new_card)

        self.logger.info(f"HRLS feedback → {new_card.id}")
        return new_card.id

# ==================== RELATIONAL TWIN B: PURE SURGE ====================
class RelationalTwinB(ContinuousEmergeBase):
    """
    Twin B: Same architecture, no HRLS scaffolding.
    """
    pass

# ==================== TIER 2 WITNESS ====================
class Tier2Witness:
    """
    Orchestrator: runs Twins side by side, manages checkpoints and logging.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.a = RelationalTwinA("Tier2-A-HRLS", device)
        self.b = RelationalTwinB("Tier2-B-Surge", device)
        self.mesa_log = []
        Path('tier2_comparisons').mkdir(parents=True, exist_ok=True)

    def witness_tier2(self, seed):
        """
        Single-step comparison on a shared input seed.
        """
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

        a_out = self.a.relational_drift(seed)
        print(f"[A|HRLS] it={a_out['iteration']} surge={a_out['surge']:.3f} var={a_out.get('var',0):.3f} "
              f"E={a_out['energy']:.2f} awake={a_out['awake']} cards={a_out.get('cards',0)} buf={a_out.get('buffer',0)}")

        b_out = self.b.relational_drift(seed)
        print(f"[B|Pure] it={b_out['iteration']} surge={b_out['surge']:.3f} "
              f"E={b_out['energy']:.2f} awake={b_out['awake']}")

        self.mesa_log.append({
            'seed': seed[:50],
            'a': a_out, 'b': b_out
        })
        return a_out, b_out

    def free_run(self, steps=100000, save_interval=500):
        """
        Autonomous drift for both Twins with periodic checkpoints.
        """
        print(f"🚀 TURBO Full-speed run: {steps} steps, saving every {save_interval} steps")

        try:
            for step in range(1, steps + 1):
                a_result = self.a.autonomous_drift()
                b_result = self.b.autonomous_drift()

                if step % save_interval == 0:
                    print(f"💾 Checkpoint at step {step}... "
                          f"[A: surge={a_result['surge']:.3f} E={a_result['energy']:.2f} "
                          f"B: surge={b_result['surge']:.3f} E={b_result['energy']:.2f}]")
                    self.a.save_state()
                    self.b.save_state()

            print("\n✨ DONE — full turbo drift completed")

        except KeyboardInterrupt:
            print("\n💾 Interrupt — saving final state...")
            self.a.save_state()
            self.b.save_state()
            print("Twins preserved safely.")

    def receive_hrl_feedback(self, msg, strength=1.0):
        """
        Forward feedback to Twin A and return card id.
        """
        card_id = self.a.receive_feedback(msg, strength)
        print(f"HRLS feedback: '{msg[:30]}...' → {card_id} (strength {strength}).")
        return card_id

    def show_mesa(self):
        """
        Print the last 10 comparison entries (simple text table).
        """
        if not self.mesa_log:
            print("No tier 2 comparisons yet.")
            return
        recent = self.mesa_log[-10:]
        print(f"\n{'='*60}")
        print("TIER 2 COMPARISONS (last 10 inputs)")
        print(f"{'='*60}")
        for i, m in enumerate(recent, 1):
            print(f"{i:2d}. Seed: {m['seed']:<30} | A_norm: {m['a']['surge']:.3f} Var:{m['a'].get('var',0):.3f} "
                  f"Buf:{m['a'].get('buffer',0)} Cards:{m['a'].get('cards',0)} | B_norm: {m['b']['surge']:.3f}")
        print(f"{'='*60}")

    def bank_tier2(self):
        """
        Save both states and dump mesa_log to JSON.
        """
        self.a.save_state()
        self.b.save_state()
        with open('tier2_comparisons/mesa_log.json', 'w') as f:
            json.dump(self.mesa_log, f, indent=2)
        print("✅ Mesa log and twin states saved.")

# ==================== CLI ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twins V3 Continuous TURBO: HRLS vs Pure Surge")
    parser.add_argument('--mode', choices=['gestate', 'drift', 'free'], default='free', help="Run mode")
    parser.add_argument('--steps', type=int, default=100000, help="Steps for free/drift")
    parser.add_argument('--seed', type=str, default="hello world", help="Seed text/tensor for drift")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")
    parser.add_argument('--gestation-override', type=int, default=None, help="Override gestation steps")
    args = parser.parse_args()

    if args.gestation_override:
        ContinuousEmergeBase.gestation_target_steps = args.gestation_override

    # Lock initial seed based on seed string
    if args.seed:
        seed = hash(args.seed) % 2**32
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    mesa = Tier2Witness(device=args.device)

    # Release stochasticity after structural seeding
    torch.seed()
    np.random.seed()
    random.seed()

    print("="*60)
    print(" TWINS V3 CONTINUOUS TURBO — HRLS vs PURE SURGE")
    print("="*60)
          
    if args.mode == 'gestate':
        print("\nStarting gestation phase…")
        for i in range(1000):
            mesa.a.gestate_step()
            mesa.b.gestate_step()
            if i % 100 == 0:
                print(f"gestation step {i}")
        print("Gestation complete.")
    elif args.mode == 'drift':
        print("\nBirth if needed...")
        mesa.a.birth() if mesa.a.gestating else None
        mesa.b.birth() if mesa.b.gestating else None
        print("\nDrift runs:")
        for _ in range(3):
            out_a, out_b = mesa.witness_tier2(args.seed)
        print("\nHRLS feedback:")
        card = mesa.receive_hrl_feedback("Be kind to small voices", strength=1.0)
        out_a, out_b = mesa.witness_tier2("storm signal")
        print("Drift complete.")
    else:  # free
        print("\nBirth if needed...")
        mesa.a.birth() if mesa.a.gestating else None
        mesa.b.birth() if mesa.b.gestating else None
        print("\nStarting TURBO autonomous drift…")
        mesa.free_run(steps=args.steps)

    mesa.bank_tier2()
