# FOV 내 다른 로봇 Trajectory를 활용한 강화학습 구현

이 문서는 다중 로봇 탐사 시뮬레이션에서 FOV(Field of View) 내에 감지된 다른 로봇의 trajectory를 강화학습에 활용하기 위한 구현 내용을 정리합니다.

## 📋 구현 개요

**목표**: 로봇이 자신의 FOV 내에 다른 로봇을 감지하면, 해당 로봇들의 최근 trajectory를 추출하여 강화학습 시 observation에 포함

**핵심 기술**:
- **Temporal Transformer**: 시계열 trajectory 데이터를 처리
- **Multi-Head Attention**: 여러 로봇의 trajectory를 aggregation
- **Positional Encoding**: 시간 정보 인코딩

---

## 🔧 주요 변경 사항

### 1. **parameter.py** - Trajectory 파라미터 추가

```python
# Trajectory tracking parameters
TRAJECTORY_HISTORY_LENGTH = 10  # Number of recent steps to track
TRAJECTORY_FEATURE_DIM = 4      # (dx, dy, heading, velocity)
TRAJECTORY_EMBEDDING_DIM = 64   # Trajectory encoder output dimension
MAX_DETECTED_AGENTS = N_AGENTS - 1  # Maximum number of detectable agents in FOV
```

**위치**: Line 80-84

---

### 2. **utils/multi_agent_worker.py** - Trajectory Buffer 구현

#### 변경 사항:
1. **Import 추가**:
   ```python
   from collections import deque
   ```

2. **Trajectory Buffer 초기화** (`__init__` 메서드):
   ```python
   # Initialize trajectory buffer for each agent
   self.trajectory_buffer = {}
   for i in range(self.n_agents):
       self.trajectory_buffer[i] = deque(maxlen=TRAJECTORY_HISTORY_LENGTH)
       # Initialize with starting positions (x, y, heading, velocity=0)
       start_location = self.env.robot_locations[i]
       self.trajectory_buffer[i].append((
           start_location[0],
           start_location[1],
           self.env.angles[i],
           0.0
       ))
   ```

3. **Trajectory 업데이트** (`run_episode` 메서드, reward 계산 부분):
   ```python
   # Update trajectory buffer
   prev_trajectory = self.trajectory_buffer[robot.id][-1] if len(self.trajectory_buffer[robot.id]) > 0 else None
   if prev_trajectory is not None:
       prev_x, prev_y = prev_trajectory[0], prev_trajectory[1]
       velocity = np.linalg.norm(next_location - np.array([prev_x, prev_y])) / NUM_SIM_STEPS
   else:
       velocity = 0.0

   self.trajectory_buffer[robot.id].append((
       next_location[0],
       next_location[1],
       robot.heading,
       velocity
   ))
   ```

4. **Observation 생성 시 trajectory_buffer 전달**:
   ```python
   # 기존 코드 (2곳):
   observation = robot.get_observation()

   # 변경 후:
   observation = robot.get_observation(
       robot_locations=self.env.robot_locations,
       trajectory_buffer=self.trajectory_buffer
   )
   ```

**위치**: Line 27, 64-75, 168-181, 91-94, 241-244

---

### 3. **utils/agent.py** - FOV 감지 및 Trajectory 추출

#### 추가된 메서드:

1. **`get_robots_in_fov(self, robot_locations)`**:
   - FOV 내에 있는 다른 로봇 감지
   - 거리 체크 (sensor_range 이내)
   - 각도 체크 (FOV 범위 내)
   - 반환: 감지된 로봇 ID 리스트

2. **`_get_detected_trajectories(self, robot_locations, trajectory_buffer)`**:
   - 감지된 로봇들의 trajectory 추출
   - 상대 좌표로 변환 및 정규화
   - Padding 처리 (MAX_DETECTED_AGENTS까지)
   - 반환: (detected_trajectories, trajectory_mask) 텐서

3. **`get_observation()` 메서드 수정**:
   ```python
   # 기존 signature:
   def get_observation(self, pad=True):

   # 변경 후:
   def get_observation(self, pad=True, robot_locations=None, trajectory_buffer=None):

   # 반환값 변경 (9개 → 11개):
   return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask,
           all_node_frontier_distribution, node_heading_visited, node_neighbor_best_headings,
           detected_trajectories, trajectory_mask]  # 2개 추가
   ```

4. **`select_next_waypoint()` 메서드 수정**:
   ```python
   # 기존:
   _, _, _, _, current_edge, _, _, _, _ = observation
   logp = self.policy_net(*observation)

   # 변경 후:
   _, _, _, _, current_edge, _, _, _, _, _, _ = observation
   logp = self.policy_net(*observation[:9], detected_trajectories=observation[9], trajectory_mask=observation[10])
   ```

5. **`save_observation()` 메서드 수정**:
   ```python
   # 기존:
   node_inputs, ..., neighbor_best_headings = observation

   # 변경 후:
   node_inputs, ..., neighbor_best_headings, detected_trajectories, trajectory_mask = observation
   # Note: detected_trajectories와 trajectory_mask는 episode_buffer에 저장하지 않음
   ```

**위치**: Line 182, 245-255, 257-260, 384-486, 510-521, 544

---

### 4. **utils/model.py** - Trajectory Encoder 및 Network 통합

#### 새로 추가된 클래스:

1. **`PositionalEncoding`**:
   - 시간 정보를 인코딩하는 Positional Encoding
   - Sinusoidal 방식 사용

2. **`TrajectoryEncoder`**:
   ```python
   class TrajectoryEncoder(nn.Module):
       def __init__(self, feature_dim, trajectory_embedding_dim, seq_len, n_head=4, n_layer=2):
           # Feature projection
           self.feature_projection = nn.Linear(feature_dim, trajectory_embedding_dim)

           # Positional encoding
           self.positional_encoding = PositionalEncoding(trajectory_embedding_dim, max_len=seq_len)

           # Temporal transformer encoder
           self.temporal_encoder = Encoder(embedding_dim=trajectory_embedding_dim, n_head=n_head, n_layer=n_layer)

           # Agent aggregation
           self.agent_attention = MultiHeadAttention(trajectory_embedding_dim, n_heads=n_head)

           # Output projection
           self.output_layer = nn.Sequential(...)
   ```

   **처리 흐름**:
   1. Input: `[batch, max_detected_agents, seq_len, feature_dim]`
   2. Feature Projection → Positional Encoding
   3. Temporal Transformer (각 agent의 trajectory 독립적으로 인코딩)
   4. Agent Aggregation (Multi-Head Attention)
   5. Output: `[batch, trajectory_embedding_dim]`

#### PolicyNet 수정:

```python
class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim, num_angles_bin, use_trajectory=True):
        # Trajectory encoder 추가
        if use_trajectory:
            self.trajectory_encoder = TrajectoryEncoder(...)
            self.trajectory_fusion = nn.Linear(embedding_dim + TRAJECTORY_EMBEDDING_DIM, embedding_dim)

    def decode_state(self, ..., trajectory_embedding=None):
        # Trajectory embedding과 current state를 fusion
        if self.use_trajectory and trajectory_embedding is not None:
            trajectory_embedding_expanded = trajectory_embedding.unsqueeze(1)
            fused = torch.cat([enhanced_current_node_feature, trajectory_embedding_expanded], dim=-1)
            enhanced_current_node_feature = self.trajectory_fusion(fused)

    def forward(self, ..., detected_trajectories=None, trajectory_mask=None):
        # Trajectory encoding
        if self.use_trajectory and detected_trajectories is not None:
            trajectory_embedding = self.trajectory_encoder(detected_trajectories, trajectory_mask)

        # Decode with trajectory fusion
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            ..., trajectory_embedding)
```

#### QNet 수정:

PolicyNet과 동일한 방식으로 수정:
- Trajectory encoder 추가
- `decode_state()` 메서드에 trajectory fusion 추가
- `forward()` 메서드에 trajectory encoding 추가

**위치**: Line 1-4 (import), 199-308 (TrajectoryEncoder), 312-427 (PolicyNet), 430-568 (QNet)

---

## 🎯 주요 특징

### 1. **Temporal Transformer Architecture**
- **시간 의존성 학습**: Positional Encoding + Self-Attention
- **병렬 처리**: 모든 agent의 trajectory를 동시에 처리
- **유연한 시퀀스 길이**: Padding으로 다양한 길이 지원

### 2. **Multi-Agent Trajectory Aggregation**
- **Cross-Attention**: 여러 로봇의 trajectory를 통합
- **Adaptive Weighting**: Attention mechanism으로 중요도 자동 학습
- **Scalable**: 감지된 로봇 수에 관계없이 고정된 출력 차원

### 3. **Feature Normalization**
- **상대 좌표**: 현재 로봇 위치 기준으로 정규화
- **범위 정규화**: 모든 feature를 [-1, 1] 또는 [0, 1] 범위로 변환
- **안정적인 학습**: Gradient 안정성 향상

---

## 📊 Trajectory Feature 구성

각 trajectory step은 4차원 feature로 표현됩니다:

```python
feature = [dx_norm, dy_norm, heading_norm, velocity_norm]
```

1. **dx_norm, dy_norm**: 상대 좌표 (현재 로봇 기준)
   - 정규화: `/ (UPDATING_MAP_SIZE / 2)`

2. **heading_norm**: 방향 (0-360도)
   - 정규화: `/ 360.0` → [0, 1]

3. **velocity_norm**: 속도
   - 정규화: `/ (VELOCITY * NUM_SIM_STEPS)`

---

## 🚀 사용 방법

### 학습 시작:
```bash
conda activate marvel
python driver.py
```

### Trajectory 기능 끄기:
```python
# utils/model.py에서
policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=False)
q_net = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO, use_trajectory=False)
```

---

## 🔍 디버깅 및 테스트

### 1. Network 초기화 테스트:
```python
from parameter import *
from utils.model import PolicyNet, QNet
import torch

policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=True)
q_net = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, TRAIN_ALGO, use_trajectory=True)
print("Networks initialized successfully!")
```

### 2. FOV 감지 테스트:
```python
from utils.agent import Agent

# agent.get_robots_in_fov(robot_locations) 호출
detected_ids = agent.get_robots_in_fov(robot_locations)
print(f"Detected robots: {detected_ids}")
```

### 3. Trajectory Buffer 확인:
```python
from utils.multi_agent_worker import MultiAgentWorker

# worker.trajectory_buffer 출력
for agent_id, trajectory in worker.trajectory_buffer.items():
    print(f"Agent {agent_id}: {len(trajectory)} steps")
```

---

## 📝 구현 시 주의사항

### 1. **Episode Buffer**
- Trajectory 정보는 **실시간으로만** 사용됩니다
- Episode buffer에는 저장되지 않음 (매번 새로 계산)
- 이유: Trajectory는 환경 상태에 의존하므로 저장 시 메모리 부담 및 재현성 문제

### 2. **Observation 구조 변경**
- 기존 9개 요소 → 11개 요소로 확장
- 기존 코드에서 observation unpacking 시 주의 필요

### 3. **Backward Compatibility**
- `use_trajectory=False`로 설정 시 기존 방식으로 동작
- 기존 체크포인트 로드 시 호환성 확인 필요

### 4. **Performance 고려사항**
- Trajectory Encoder는 추가 연산 비용 발생
- GPU 사용 권장 (`USE_GPU_GLOBAL = True`)

---

## 🎨 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Multi-Robot Environment                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Robot 0 │  │ Robot 1 │  │ Robot 2 │  │ Robot 3 │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
└───────┼───────────┼────────────┼─────────────┼─────────────┘
        │           │            │             │
        └───────────┴────────────┴─────────────┘
                     │
              ┌──────▼──────┐
              │ FOV Detection│
              │ (Agent.get_  │
              │ robots_in_   │
              │ fov())       │
              └──────┬───────┘
                     │
        ┌────────────▼────────────┐
        │ Trajectory Extraction   │
        │ (Recent N steps)        │
        │ - Position (x, y)       │
        │ - Heading (θ)           │
        │ - Velocity (v)          │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Feature Projection    │
        │   + Positional Encoding │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ Temporal Transformer    │
        │ (Self-Attention over    │
        │  time steps)            │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Agent Aggregation      │
        │  (Cross-Attention)      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │  Trajectory Embedding   │
        │  [batch, 64]            │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   State Fusion          │
        │   (Concat + Linear)     │
        │                         │
        │   Current State ────┐   │
        │                     │   │
        │   Trajectory ───────┴──►│
        │   Embedding            │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │   Policy / Q-Network    │
        │   Action Selection      │
        └─────────────────────────┘
```

---

## 🧪 테스트 코드 수정 사항

test_driver.py를 통해 학습된 모델을 테스트할 때도 trajectory encoder가 반영되도록 다음 파일들을 수정했습니다:

### 1. **test_parameter.py** 수정

```python
# Network parameters
USE_TRAJECTORY = True  # Enable trajectory encoder

# Trajectory tracking parameters (same as parameter.py)
TRAJECTORY_HISTORY_LENGTH = 10
TRAJECTORY_FEATURE_DIM = 4
TRAJECTORY_EMBEDDING_DIM = 64
MAX_DETECTED_AGENTS = 10  # Conservative estimate for test
```

**위치**: Line 59-66

### 2. **test_driver.py** 수정

#### Runner 클래스의 network 초기화:
```python
# 기존:
self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)

# 변경 후:
self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=USE_TRAJECTORY)
```

**위치**: Line 128

### 3. **utils/test_worker.py** 수정

#### Import 추가:
```python
from collections import deque
```

#### Trajectory Buffer 초기화 (`__init__` 메서드):
```python
# Initialize trajectory buffer for each agent
self.trajectory_buffer = {}
for i in range(self.n_agents):
    self.trajectory_buffer[i] = deque(maxlen=TRAJECTORY_HISTORY_LENGTH)
    start_location = self.env.robot_locations[i]
    self.trajectory_buffer[i].append((
        start_location[0],
        start_location[1],
        self.env.angles[i],
        0.0
    ))
```

**위치**: Line 6, 42-53

#### Observation 생성 시 trajectory_buffer 전달:
```python
# 기존:
observation = robot.get_observation(pad=False)

# 변경 후:
observation = robot.get_observation(
    pad=False,
    robot_locations=self.env.robot_locations,
    trajectory_buffer=self.trajectory_buffer
)
```

**위치**: Line 83-87

#### Trajectory 업데이트:
```python
# Update trajectory buffer
prev_trajectory = self.trajectory_buffer[robot.id][-1] if len(self.trajectory_buffer[robot.id]) > 0 else None
if prev_trajectory is not None:
    prev_x, prev_y = prev_trajectory[0], prev_trajectory[1]
    velocity = np.linalg.norm(next_location - np.array([prev_x, prev_y])) / NUM_SIM_STEPS
else:
    velocity = 0.0

self.trajectory_buffer[robot.id].append((
    next_location[0],
    next_location[1],
    robot.heading,
    velocity
))
```

**위치**: Line 159-172

#### `__main__` 부분 수정:
```python
# 기존:
policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)

# 변경 후:
policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=USE_TRAJECTORY)
```

**위치**: Line 377

### 테스트 실행 방법:

```bash
conda activate marvel
python test_driver.py
```

### 주의사항:

1. **체크포인트 호환성**:
   - Trajectory encoder가 포함된 모델로 학습된 체크포인트를 사용해야 합니다
   - 기존 체크포인트를 사용하려면 `USE_TRAJECTORY = False`로 설정하세요

2. **동적 Agent 수**:
   - Test에서는 agent 수가 동적으로 변경될 수 있습니다
   - `MAX_DETECTED_AGENTS`를 충분히 크게 설정했습니다 (10)

3. **성능 비교**:
   - Trajectory 기능 ON/OFF로 성능 비교 가능
   - `test_parameter.py`의 `USE_TRAJECTORY` 플래그로 제어

---

## 🎨 향상된 시각화 기능

test_driver.py를 통해 생성되는 GIF에 FOV 내 감지된 로봇의 trajectory를 시각적으로 표시합니다.

### 시각화 특징:

#### **왼쪽 패널 (Belief Map + Trajectories)**:
1. **기본 Trajectory**: 모든 로봇의 trajectory를 반투명하게 표시 (alpha=0.4)
2. **감지된 Trajectory 강조**:
   - 다른 로봇의 FOV에 감지된 로봇의 trajectory를 두껍고 점선으로 표시
   - 현재 위치에 흰색 테두리의 원형 마커 추가
   - linewidth=3.0, linestyle='--'

#### **오른쪽 패널 (FOV Cones + Detection Links)**:
1. **FOV Cone**: 각 로봇의 시야 범위를 부채꼴로 표시
2. **감지된 Trajectory 강조**: 왼쪽 패널과 동일
3. **Detection Links**: 감지하는 로봇과 감지된 로봇 사이를 흰색 점선으로 연결
4. **Detection Summary**: 제목에 각 로봇이 감지한 다른 로봇들을 텍스트로 표시

### 구현 코드:

```python
def get_detected_robots_in_fov(self, robot, robot_locations, robot_headings):
    """Helper function to detect which robots are in the FOV of a given robot"""
    detected_robots = []
    robot_loc = get_coords_from_cell_position(robot_locations[robot.id], self.env.belief_info)

    for other_robot in self.robot_list:
        if other_robot.id == robot.id:
            continue

        other_loc = get_coords_from_cell_position(robot_locations[other_robot.id], self.env.belief_info)

        # Calculate distance
        distance = np.linalg.norm(other_loc - robot_loc)

        # Check if within sensor range
        if distance > self.sensor_range:
            continue

        # Calculate angle to the other robot
        delta = other_loc - robot_loc
        angle_to_robot = np.degrees(np.arctan2(delta[1], delta[0])) % 360

        # Calculate angle difference considering FOV
        angle_diff = (angle_to_robot - robot_headings[robot.id] + 180) % 360 - 180

        # Check if within FOV
        if np.abs(angle_diff) <= self.fov / 2:
            detected_robots.append(other_robot.id)

    return detected_robots
```

### 시각화 예시:

```
Title: Explored: 0.85  Distance: 45.2
       Headings: Red-90°, Blue-45°, Green-180°, Yellow-270°
       FOV Detections: Red detects: Blue, Green | Blue detects: Red

[Left Panel]                    [Right Panel]
- All trajectories (faded)      - All trajectories (faded)
- Detected: Blue (thick dash)   - Detected: Blue (thick dash)
- Detected: Green (thick dash)  - Detection links (white dash)
- Detected: Red (thick dash)    - FOV cones (semi-transparent)
```

### 시각적 요소:

| 요소 | 스타일 | 의미 |
|------|--------|------|
| 일반 Trajectory | 가는 실선, alpha=0.4 | 모든 로봇의 이동 경로 |
| 감지된 Trajectory | 두꺼운 점선, alpha=1.0 | 다른 로봇의 FOV에 포착된 경로 |
| 현재 위치 마커 | 흰색 테두리 원 | 감지된 로봇의 현재 위치 |
| Detection Link | 흰색 점선 | 감지 관계 연결선 |
| FOV Cone | 부채꼴, alpha=0.3 | 로봇의 시야 범위 |

---

## 📚 참고 자료

### Training 관련 파일:
- `parameter.py`: Line 80-84 (Trajectory parameters)
- `driver.py`: PolicyNet/QNet 초기화 부분
- `utils/multi_agent_worker.py`: Line 27, 64-75, 168-181, 91-94, 241-244
- `utils/agent.py`: Line 182, 245-255, 257-260, 384-486, 510-521, 544
- `utils/model.py`: Line 1-4, 199-308, 312-427, 430-568

### Testing 관련 파일:
- `test_parameter.py`: Line 59-66 (Trajectory parameters)
- `test_driver.py`: Line 40-45 (global_network), Line 128 (Runner.local_network)
- `utils/test_worker.py`: Line 6, 42-53, 83-87, 159-172, 377
  - Line 284-313: `get_detected_robots_in_fov()` - FOV 내 로봇 감지 함수
  - Line 315-474: `plot_local_env_sim()` - 향상된 시각화 (감지된 trajectory 강조)

### 핵심 함수:
- `MultiAgentWorker.__init__()`: Trajectory buffer 초기화
- `Agent.get_robots_in_fov()`: FOV 내 로봇 감지
- `Agent._get_detected_trajectories()`: Trajectory 추출 및 인코딩
- `TrajectoryEncoder.forward()`: Temporal transformer 처리
- `PolicyNet.decode_state()`: Trajectory fusion

---

## ✅ 체크리스트

구현 완료 항목:
- [x] Trajectory 파라미터 추가 (parameter.py)
- [x] Trajectory buffer 구현 (multi_agent_worker.py)
- [x] FOV 감지 함수 (agent.py)
- [x] Trajectory Encoder with Transformer (model.py)
- [x] PolicyNet 통합 (model.py)
- [x] QNet 통합 (model.py)
- [x] Observation 생성 업데이트 (agent.py)
- [ ] 실제 학습 테스트
- [ ] 성능 평가 및 비교

---

## 💡 향후 개선 방향

1. **Trajectory 예측**: 미래 trajectory 예측 기능 추가
2. **Communication**: 명시적 agent 간 communication channel
3. **Hierarchical Attention**: 시간/공간 계층적 attention
4. **Memory Module**: Long-term trajectory memory
5. **Adaptive History Length**: 동적 history 길이 조정

---

**작성일**: 2025-11-25
**버전**: 1.0
**작성자**: Claude (Anthropic)
