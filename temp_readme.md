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

test_driver.py를 통해 생성되는 GIF에 FOV 내 감지된 로봇의 trajectory와 각 agent의 local view를 시각적으로 표시합니다.

### 전체 레이아웃 (2행 구조):

#### **상단 행 (Global View)**:
1. **왼쪽 패널 - Global Belief Map**:
   - 모든 로봇의 trajectory를 반투명하게 표시 (alpha=0.4)
   - 감지된 로봇의 trajectory를 두껍고 점선으로 강조 (linewidth=3.0, linestyle='--')
   - 현재 위치에 흰색 테두리의 원형 마커 추가
   - Global frontiers 표시

2. **오른쪽 패널 - FOV & Detections**:
   - FOV Cone: 각 로봇의 시야 범위를 부채꼴로 표시
   - Detection Links: 감지하는 로봇과 감지된 로봇 사이를 흰색 점선으로 연결
   - 감지된 Trajectory 강조
   - Detection Summary: 제목에 각 로봇이 감지한 다른 로봇들을 텍스트로 표시

#### **하단 행 (Individual Agent Local Views)** - 🆕 새로 추가:
각 agent마다 개별 패널로 자신의 local observation을 표시:

1. **Local Map**:
   - 각 agent의 현재 위치 중심으로 UPDATING_MAP_SIZE 범위 내 지도 표시
   - Agent가 실제로 decision-making에 사용하는 local view 시각화

2. **FOV Cone**:
   - Agent의 시야 범위를 반투명 부채꼴로 표시
   - Agent가 현재 어느 방향을 보고 있는지 명확히 표시

3. **Detected Robots (FOV 내 감지된 다른 로봇)**:
   - **감지된 로봇**: 큰 원형 마커 + 노란색 테두리 (markeredgecolor='yellow')
   - **Detection Line**: Agent와 감지된 로봇 사이 노란색 점선으로 연결
   - **비감지 로봇**: 작고 반투명한 마커 (alpha=0.5)

4. **Local Frontiers**:
   - Agent가 관측하는 frontiers를 빨간 점으로 표시
   - Agent의 exploration 목표 지점 시각화

5. **Title**:
   - Agent 이름 (색상과 함께)
   - 현재 감지 중인 다른 robot 목록 표시

### 시각화 코드 구현:

**위치**: utils/test_worker.py, Line 315-589

#### 주요 기능:

1. **레이아웃 자동 조정** (Line 318-320):
   ```python
   n_cols = max(2, self.n_agents)
   fig = plt.figure(figsize=(3 * n_cols, 6))
   ```
   - Agent 수에 따라 열 개수 자동 조정
   - 상단 2개 패널, 하단 agent 수만큼 패널 생성

2. **Local Map 추출** (Line 477-486):
   ```python
   center_cell = robot_locations[robot.id]
   half_size = local_map_size // 2

   row_start = max(0, int(center_cell[1] - half_size))
   row_end = min(self.env.robot_belief.shape[0], int(center_cell[1] + half_size))
   col_start = max(0, int(center_cell[0] - half_size))
   col_end = min(self.env.robot_belief.shape[1], int(center_cell[0] + half_size))

   local_map = self.env.robot_belief[row_start:row_end, col_start:col_end]
   ```

3. **FOV 내 다른 Robot 시각화** (Line 519-547):
   ```python
   # Check if this other robot is detected by current robot
   is_detected = other_robot.id in fov_detections.get(robot.id, [])

   if is_detected:
       # Highlight detected robots with yellow border
       plt.plot(other_local_x, other_local_y, 'o',
               color=other_c, markersize=10,
               markeredgewidth=3, markeredgecolor='yellow', zorder=15)
       # Draw detection line
       plt.plot([robot_local_x, other_local_x], [robot_local_y, other_local_y],
               'y--', linewidth=2, alpha=0.8, zorder=12)
   ```

### 시각화 예시:

```
Title: Explored: 0.85  Distance: 45.2
       Headings: Red-90°, Blue-45°, Green-180°, Yellow-270°
       FOV Detections: Red detects: Blue, Green | Blue detects: Red

┌─────────────────────────────────────────────────────────────┐
│  [Global Belief Map]       [FOV & Detections]              │
│  - All trajectories        - FOV cones                      │
│  - Detected highlighted    - Detection links                │
│  - Global frontiers        - Highlighted detections         │
├─────────────────────────────────────────────────────────────┤
│  [Red Agent Local]  [Blue Agent Local]  [Green] [Yellow]   │
│  - Local map        - Local map         - ...   - ...      │
│  - FOV cone         - FOV cone                              │
│  - Detected: Blue   - Detected: Red                         │
│  - Local frontiers  - Local frontiers                       │
└─────────────────────────────────────────────────────────────┘
```

### 시각적 요소:

| 요소 | 스타일 | 의미 |
|------|--------|------|
| **Global View** | | |
| 일반 Trajectory | 가는 실선, alpha=0.4 | 모든 로봇의 이동 경로 |
| 감지된 Trajectory | 두꺼운 점선, alpha=1.0 | 다른 로봇의 FOV에 포착된 경로 |
| 현재 위치 마커 | 흰색 테두리 원 | 감지된 로봇의 현재 위치 |
| Detection Link | 흰색 점선 | 감지 관계 연결선 |
| FOV Cone | 부채꼴, alpha=0.3 | 로봇의 시야 범위 |
| **Local View (각 Agent)** | | |
| Local Map | UPDATING_MAP_SIZE 범위 | Agent의 decision-making 영역 |
| 감지된 로봇 | 큰 원 + 노란색 테두리 | FOV 내에서 감지된 다른 로봇 |
| Detection Line | 노란색 점선 | Agent와 감지된 로봇 간 연결 |
| 비감지 로봇 | 작은 원, alpha=0.5 | Local 영역 내 비감지 로봇 |
| Local Frontiers | 빨간 점, s=2 | Agent가 관측하는 frontier |

### 실제 활용:

이 시각화를 통해 다음을 확인할 수 있습니다:

1. **통신 없는 학습 검증**:
   - 각 agent가 독립적인 local observation만 사용하는지 확인
   - FOV 밖의 로봇은 감지되지 않음을 시각적으로 확인

2. **Trajectory Encoder 효과**:
   - 각 agent가 어떤 다른 로봇을 감지하고 있는지 명확히 표시
   - Detection line으로 information flow 시각화

3. **Decision-making 분석**:
   - 각 agent의 local frontiers와 선택한 경로 관찰
   - Agent가 감지된 로봇을 피하거나 협력하는 행동 분석

4. **성능 디버깅**:
   - Agent가 frontier를 제대로 감지하는지 확인
   - FOV 범위가 올바르게 적용되는지 검증

---

## 📚 참고 자료

### Training 관련 파일:
- `parameter.py`: Line 80-84 (Trajectory parameters)
- `driver.py`: PolicyNet/QNet 초기화 부분
- `utils/multi_agent_worker.py`: Line 27, 64-75, 168-181, 91-94, 241-244
- `utils/agent.py`: Line 182, 245-255, 257-260, 384-486, 510-521, 544
- `utils/model.py`: Line 1-4, 199-308, 312-427, 430-568

### Testing 관련 파일:
- `test_parameter.py`:
  - Line 59-66: Trajectory parameters
  - Line 77-79: Communication settings
- `test_driver.py`: Line 40-45 (global_network), Line 128 (Runner.local_network)
- `utils/test_worker.py`:
  - Line 6, 42-53: Trajectory buffer 초기화
  - Line 83-87, 159-172: Trajectory buffer 업데이트 및 사용
  - Line 284-313: `get_detected_robots_in_fov()` - FOV 내 로봇 감지 함수
  - Line 315-589: `plot_local_env_sim()` - 향상된 시각화
    - Line 318-320: 2행 레이아웃 구조 (상단: global view, 하단: per-agent local views)
    - Line 331-392: Global belief map 패널
    - Line 394-465: FOV & detections 패널
    - Line 467-567: **각 Agent별 local view 패널 (새로 추가)**

### 핵심 함수:
- `MultiAgentWorker.__init__()`: Trajectory buffer 초기화
- `Agent.get_robots_in_fov()`: FOV 내 로봇 감지
- `Agent._get_detected_trajectories()`: Trajectory 추출 및 인코딩
- `TrajectoryEncoder.forward()`: Temporal transformer 처리
- `PolicyNet.decode_state()`: Trajectory fusion

---

## ✅ 체크리스트

구현 완료 항목:

### Trajectory Encoder 구현:
- [x] Trajectory 파라미터 추가 (parameter.py)
- [x] Trajectory buffer 구현 (multi_agent_worker.py)
- [x] FOV 감지 함수 (agent.py)
- [x] Trajectory Encoder with Transformer (model.py)
- [x] PolicyNet 통합 (model.py)
- [x] QNet 통합 (model.py)
- [x] Observation 생성 업데이트 (agent.py)

### 통신 설정 구현:
- [x] USE_COMMUNICATION 파라미터 추가 (parameter.py, test_parameter.py)
- [x] effective_train_algo 로직 구현 (driver.py)
- [x] 조건부 agent indices 저장 (multi_agent_worker.py)
- [x] 학습 루프 state 구성 수정 (driver.py)
- [x] 문서화 (temp_readme.md)

### 향상된 시각화 구현:
- [x] FOV 내 감지된 trajectory 강조 (test_worker.py)
- [x] Detection links 표시 (test_worker.py)
- [x] **각 Agent별 local view 추가 (test_worker.py)** 🆕
- [x] Local map 추출 및 표시
- [x] FOV cone 시각화
- [x] Detected robots 강조 표시
- [x] Local frontiers 표시

### 테스트 및 평가:
- [ ] 실제 학습 테스트
- [ ] 통신 있음/없음 성능 비교
- [ ] Local view 시각화 검증

---

## 🔌 통신 설정 (Communication Settings)

MARVEL은 이제 에이전트 간 통신 여부를 제어할 수 있습니다. 이를 통해 완전한 정보 공유(centralized) vs 시각적 감지만 사용(decentralized) 두 가지 학습 모드를 지원합니다.

### USE_COMMUNICATION 파라미터

**parameter.py** (Line 100-104):
```python
USE_COMMUNICATION = False  # True: MAAC with all agent communication (centralized critic)
                           # False: Decentralized learning with only FOV-based trajectory observation
                           # When False, agents only use their own observation + detected trajectories in FOV
                           # This simulates no-communication scenario where agents rely on visual detection only
```

### 통신 모드별 차이점

#### 1. **USE_COMMUNICATION = True** (통신 있음)
- **학습 알고리즘**: TRAIN_ALGO에 따라 MAAC 또는 MAAC+GT 사용
- **정보 공유**: 모든 에이전트의 위치와 상태 정보를 QNet에 전달
- **Centralized Critic**: 글로벌 정보를 활용한 가치 평가
- **장점**: 더 많은 정보로 학습, 수렴 속도 빠름
- **단점**: 실제 환경에서 통신 인프라 필요

#### 2. **USE_COMMUNICATION = False** (통신 없음)
- **학습 알고리즘**: TRAIN_ALGO에서 통신 요소 제거
  - TRAIN_ALGO 3 (MAAC+GT) → effective_train_algo 2 (GT only)
  - TRAIN_ALGO 1 (MAAC) → effective_train_algo 0 (SAC)
- **정보 공유**: 없음 (각 에이전트가 독립적으로 학습)
- **시각적 감지**: FOV 내 감지된 로봇의 trajectory만 사용
- **장점**: 통신 인프라 불필요, 더 현실적인 시나리오
- **단점**: 제한된 정보로 학습, 수렴 속도 느릴 수 있음

### 구현 상세

#### 1. **driver.py** - Effective Training Algorithm 계산

**위치**: Line 53-72

```python
# Determine effective training algorithm based on communication setting
# When USE_COMMUNICATION=False, disable agent communication in QNet
if USE_COMMUNICATION:
    effective_train_algo = TRAIN_ALGO
else:
    # Remove agent communication component from TRAIN_ALGO
    # TRAIN_ALGO 3 (MAAC + GT) -> 2 (GT only)
    # TRAIN_ALGO 1 (MAAC) -> 0 (SAC)
    if TRAIN_ALGO == 3:
        effective_train_algo = 2  # Ground Truth only, no communication
    elif TRAIN_ALGO == 1:
        effective_train_algo = 0  # SAC, no communication
    else:
        effective_train_algo = TRAIN_ALGO  # 0 or 2 already have no communication

print(f"Training Configuration:")
print(f"  TRAIN_ALGO: {TRAIN_ALGO}")
print(f"  USE_COMMUNICATION: {USE_COMMUNICATION}")
print(f"  Effective TRAIN_ALGO for QNet: {effective_train_algo}")
print(f"  Using Trajectory Encoder: True")
```

**Network 초기화** (Line 75-82):
```python
global_policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, use_trajectory=True).to(device)
global_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True).to(device)
global_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True).to(device)
# ...
global_target_q_net1 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, effective_train_algo, use_trajectory=True).to(device)
global_target_q_net2 = QNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_AGES_BIN, effective_train_algo, use_trajectory=True).to(device)
```

#### 2. **multi_agent_worker.py** - 조건부 Agent Indices 저장

**위치**: Line 224-232

```python
curr_node_indices = np.array([robot.current_index for robot in self.robot_list])
for robot, reward in zip(self.robot_list, reward_list):
    robot.save_reward(reward + team_reward)
    # Only save all agent indices when communication is enabled
    # When USE_COMMUNICATION=False, agents rely solely on FOV-detected trajectories
    if USE_COMMUNICATION:
        robot.save_all_indices(curr_node_indices)
    robot.update_planning_state(self.env.robot_locations)
    robot.save_done(done)
```

**핵심**: `USE_COMMUNICATION=False`일 때는 `save_all_indices()`를 호출하지 않아, episode buffer에 다른 에이전트의 위치 정보가 저장되지 않습니다.

#### 2-1. **agent.py** - save_next_observations 수정

**위치**: Line 532-563

```python
def save_next_observations(self, observation, next_node_index_list):
    # ... (기존 코드)

    # Only process agent indices if they were saved (USE_COMMUNICATION=True)
    if len(self.episode_buffer[35]) > 0:
        self.episode_buffer[36] = copy.deepcopy(self.episode_buffer[35])[1:]

    # ... (observation 저장)

    # Only update agent indices buffers if they were initialized
    if len(self.episode_buffer[35]) > 0:
        self.episode_buffer[36] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
        self.episode_buffer[37] = copy.deepcopy(self.episode_buffer[36])[1:]
        self.episode_buffer[37] += copy.deepcopy(self.episode_buffer[36])[-1:]
```

**핵심**: episode_buffer[35]가 비어있을 때 (USE_COMMUNICATION=False) episode_buffer[36], [37]을 처리하지 않아 IndexError를 방지합니다.

#### 3. **driver.py** - 빈 버퍼 처리

**위치**: Line 200-212

```python
indices = range(len(experience_buffer[0]))

# training for n times each step
for j in range(4):
    # randomly sample a batch data
    sample_indices = random.sample(indices, BATCH_SIZE)
    rollouts = []
    for i in range(len(experience_buffer)):
        # Skip empty buffers (e.g., agent indices when USE_COMMUNICATION=False)
        if len(experience_buffer[i]) == 0:
            rollouts.append([])
        else:
            rollouts.append([experience_buffer[i][index] for index in sample_indices])
```

**핵심**: experience_buffer의 일부(35, 36, 37)가 비어있을 때 IndexError를 방지합니다.

#### 4. **driver.py** - 학습 루프에서 State 구성

**위치**: Line 233-285

**Ground Truth 데이터 로딩** (Line 233-250):
```python
# Load ground truth data if needed
if effective_train_algo in (2,3):
    gt_node_inputs = torch.stack(rollouts[19]).to(device)
    # ... (ground truth data loading)
```

**Agent Indices 로딩** (Line 252-257):
```python
# Load agent indices only when communication is enabled
# When USE_COMMUNICATION=False, effective_train_algo won't include agent communication
if effective_train_algo in (1,3):
    all_agent_indices = torch.stack(rollouts[35]).to(device)
    all_agent_next_indices = torch.stack(rollouts[36]).to(device)
    next_all_agent_next_indices = torch.stack(rollouts[37]).to(device)
```

**State 구성** (Line 264-285):
```python
# Construct state based on effective_train_algo (respects USE_COMMUNICATION setting)
if effective_train_algo == 0:
    # SAC: observation only, no communication
    state = observation
    next_state = next_observation
elif effective_train_algo == 1:
    # MAAC with communication: observation + agent indices
    state = [*observation, all_agent_indices, all_agent_next_indices]
    next_state = [*next_observation, all_agent_next_indices, next_all_agent_next_indices]
elif effective_train_algo == 2:
    # Ground truth only, no communication
    state = [gt_node_inputs, gt_node_padding_mask, ...]
    next_state = [gt_next_node_inputs, ...]
elif effective_train_algo == 3:
    # MAAC with ground truth and communication
    state = [gt_node_inputs, ..., all_agent_indices, all_agent_next_indices]
    next_state = [gt_next_node_inputs, ..., all_agent_next_indices, next_all_agent_next_indices]
```

### TRAIN_ALGO와 USE_COMMUNICATION 조합

| TRAIN_ALGO | USE_COMMUNICATION | effective_train_algo | 설명 |
|------------|-------------------|---------------------|------|
| 0 (SAC) | True | 0 | SAC, no communication |
| 0 (SAC) | False | 0 | SAC, no communication |
| 1 (MAAC) | True | 1 | MAAC with communication |
| 1 (MAAC) | False | 0 | SAC, FOV trajectory only |
| 2 (GT) | True | 2 | Ground Truth, no communication |
| 2 (GT) | False | 2 | Ground Truth, no communication |
| 3 (MAAC+GT) | True | 3 | MAAC+GT with communication |
| 3 (MAAC+GT) | False | 2 | GT only, FOV trajectory only |

### 사용 예시

#### 통신 없는 현실적 시나리오 학습:
```python
# parameter.py
N_AGENTS = 4
USE_COMMUNICATION = False
TRAIN_ALGO = 3  # Will use GT only (effective_train_algo=2)
USE_CONTINUOUS_SIM = True

# Trajectory settings
TRAJECTORY_HISTORY_LENGTH = 10
TRAJECTORY_EMBEDDING_DIM = 64
MAX_DETECTED_AGENTS = 3  # N_AGENTS - 1
```

```bash
conda activate marvel
python driver.py
```

출력 예시:
```
Training Configuration:
  TRAIN_ALGO: 3
  USE_COMMUNICATION: False
  Effective TRAIN_ALGO for QNet: 2
  Using Trajectory Encoder: True
```

#### 통신 있는 Centralized 학습:
```python
# parameter.py
N_AGENTS = 4
USE_COMMUNICATION = True
TRAIN_ALGO = 3  # Will use MAAC+GT (effective_train_algo=3)
```

```bash
python driver.py
```

출력 예시:
```
Training Configuration:
  TRAIN_ALGO: 3
  USE_COMMUNICATION: True
  Effective TRAIN_ALGO for QNet: 3
  Using Trajectory Encoder: True
```

### 성능 비교 실험

두 모드의 성능을 비교하려면:

1. **통신 있는 모델 학습**:
   ```python
   # parameter.py
   FOLDER_NAME = 'with_communication'
   USE_COMMUNICATION = True
   TRAIN_ALGO = 3
   ```

2. **통신 없는 모델 학습**:
   ```python
   # parameter.py
   FOLDER_NAME = 'no_communication'
   USE_COMMUNICATION = False
   TRAIN_ALGO = 3
   ```

3. **TensorBoard로 비교**:
   ```bash
   tensorboard --logdir train/
   ```

### 주의사항

1. **체크포인트 호환성**:
   - `USE_COMMUNICATION`이 다른 설정으로 학습된 모델은 다른 QNet 구조를 가집니다
   - 체크포인트 로드 시 동일한 `USE_COMMUNICATION` 설정 필요

2. **Episode Buffer**:
   - `USE_COMMUNICATION=False`일 때는 agent indices가 episode buffer에 저장되지 않음
   - 메모리 사용량 감소 효과

3. **Trajectory Encoder의 역할**:
   - `USE_COMMUNICATION=False`일 때 trajectory encoder가 더욱 중요
   - FOV 감지된 로봇의 정보가 유일한 다른 에이전트 정보원

### Test Mode에서의 통신 설정

**test_parameter.py** (Line 77-79):
```python
# Communication settings (same as parameter.py)
USE_COMMUNICATION = False  # True: Use all agent communication
                           # False: Decentralized testing with only FOV-based trajectory observation
```

**중요**: Test mode에서는 PolicyNet만 사용하므로 agent 간 통신 정보가 자동으로 제외됩니다:
- **Training**: PolicyNet (actor) + QNet (critic)
  - QNet이 agent indices를 사용 (USE_COMMUNICATION=True일 때만)
- **Testing**: PolicyNet만 사용
  - PolicyNet은 observation만 사용 (agent indices 불필요)
  - 따라서 test에서는 항상 통신 없이 동작

**실제 효과**:
- Test에서는 각 agent가 자신의 observation + FOV 내 감지된 trajectory만으로 행동 결정
- 다른 agent의 전역 위치 정보는 사용하지 않음
- Training에서 `USE_COMMUNICATION=False`로 학습된 모델이 test에서 올바르게 평가됨

---

## 💡 향후 개선 방향

1. **Trajectory 예측**: 미래 trajectory 예측 기능 추가
2. **Partial Communication**: 제한적 통신 시나리오 (거리 기반, 대역폭 제한)
3. **Hierarchical Attention**: 시간/공간 계층적 attention
4. **Memory Module**: Long-term trajectory memory
5. **Adaptive History Length**: 동적 history 길이 조정

---

**작성일**: 2025-11-26
**버전**: 1.2
**작성자**: Claude (Anthropic)
**최근 업데이트**: 각 Agent별 local view 시각화 추가 (test_worker.py)
