import tensorflow as tf
from tensorflow import keras

from tf_agents.environments import py_environment, tf_py_environment, wrappers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import network, q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver

import numpy as np


# 毎回同じ結果にするための設定
# import random
# seed = 1
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# 環境の設定
class EnvironmentSimulator(py_environment.PyEnvironment):
    # 初期化
    def __init__(self):
        super(EnvironmentSimulator, self).__init__()
        # 状態の設定
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=[0, 0], maximum=[1, 1]  # 次元数，タイプ，最小値，最大値
        )
        # 行動の設定
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1
        )
        # 状態を初期値に戻すための関数の呼び出し
        self._reset()

    # 状態のリストを戻す関数（この本では変更しない）
    def observation_spec(self):
        return self._observation_spec

    # 行動のリストを戻す関数（この本では変更しない）
    def action_spec(self):
        return self._action_spec

    # 状態を初期値に戻すための関数
    def _reset(self):
        self._state = [0, 1]  # 桶：下，水：有
        return ts.restart(np.array(self._state, dtype=np.int32))

    # 行動の関数
    def _step(self, action):
        next_state = self._state.copy()
        reward = 0
        # 行動による状態遷移
        if self._state[0] == 0 and self._state[1] == 1:  # 桶：下，水：有
            if action == 0:  # 紐を引く
                next_state[0] = 1  # 桶が上になる
        elif self._state[0] == 1 and self._state[1] == 1:  # 桶：上，水：有
            if action == 0:  # 紐を引く
                next_state[0] = 0  # 桶が下になる
            elif action == 1:  # 桶を傾ける
                next_state[1] = 0  # 水がなくなる
                reward = 1  # 【報酬を得る】
        elif self._state[0] == 1 and self._state[1] == 0:  # 桶：上，水：無
            if action == 0:  # 紐を引く
                next_state[0] = 0  # 桶が下になる
                next_state[1] = 1  # 水が入る
        # 状態を更新
        self._state = next_state
        # 戻り値の設定
        return ts.transition(np.array(self._state, dtype=np.int32), reward=reward, discount=1)


# エージェントの設定
class MyQNetwork(network.Network):
    # 初期化
    def __init__(self, observation_spec, action_spec, name='QNetwork'):
        q_network.validate_specs(action_spec, observation_spec)
        n_action = action_spec.maximum - action_spec.minimum + 1
        super(MyQNetwork, self).__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            name=name
        )
        # ネットワークの設定
        self.model = keras.Sequential(
            [
                keras.layers.Dense(10, activation='relu'),
                keras.layers.Dense(10, activation='relu'),
                keras.layers.Dense(n_action),
            ]
        )

    # モデルを戻す関数（この本ではほぼ変更しない）
    def call(self, observation, step_type=None, network_state=(), training=True):
        return self.model(observation, training=training), network_state


# メイン関数
def main():
    # 環境の設定
    env = tf_py_environment.TFPyEnvironment(
        wrappers.TimeLimit(
            env=EnvironmentSimulator(),
            duration=15  # 1エピソードで行われる行動の数
        )
    )
    # ネットワークの設定
    primary_network = MyQNetwork(
        env.observation_spec(),
        env.action_spec()
    )
    # ネットワークの概要の出力（必要ない場合はコメントアウト）
    # primary_network.build(input_shape=(None,*(env.observation_spec().shape)))
    # primary_network.model.summary()
    # エージェントの設定
    n_step_update = 1
    agent = dqn_agent.DdqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=primary_network,  # 設定したネットワーク
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),  # 最適化関数
        n_step_update=n_step_update,  # 更新頻度
        epsilon_greedy=1.0,
        target_update_tau=1.0,  # 更新する頻度を設定する係数
        target_update_period=10,  # どのくらい前のネットワークを用いて更新するかの設定
        gamma=0.8,  # Q値の更新のためのパラメータ
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=tf.Variable(0)
    )
    # エージェントの初期化
    agent.initialize()
    agent.train = common.function(agent.train)
    # エージェントの行動の設定（ポリシーの設定）
    policy = agent.collect_policy
    # データの記録の設定
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,  # バッチサイズ
        max_length=10 ** 6
    )
    # TensorFlow学習用のオブジェクトへの整形
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=32,
        num_steps=n_step_update + 1
    ).prefetch(3)
    # データ形式の整形
    iterator = iter(dataset)
    # replay_bufferの自動更新の設定
    driver = dynamic_step_driver.DynamicStepDriver(
        env,
        policy,
        observers=[replay_buffer.add_batch],
    )
    driver.run(maximum_iterations=50)
    # 変数の設定
    num_episodes = 50  # エピソード数
    line_epsilon = np.linspace(start=1, stop=0, num=num_episodes)  # ランダム行動の確率
    # エピソードの繰り返し
    for episode in range(num_episodes):
        episode_rewards = 0  # 1エピソード中の報酬の合計値の初期化
        episode_average_loss = []  # 平均lossの初期化

        time_step = env.reset()  # エージェントの初期化
        policy._epsilon = line_epsilon[episode]  # ランダム行動の確率の設定
        # 設定した行動回数の繰り返し
        while True:
            policy_step = policy.action(time_step)  # 現在の状態から次の行動
            next_time_step = env.step(policy_step.action)  # 行動から次の状態
            # エピソードの保存
            traj = trajectory.from_transition(time_step, policy_step, next_time_step)
            replay_buffer.add_batch(traj)
            # 実行状態の表示（学習には関係しない）
            S = time_step.observation.numpy().tolist()[0]  # 現在の状態
            A = policy_step.action.numpy().tolist()[0]  # 行動
            R = next_time_step.reward.numpy().astype('int').tolist()[0]  # 報酬
            S_ = next_time_step.observation.numpy().tolist()[0]  # 次の状態
            print(S, A, R, S_)
            # 学習
            experience, _ = next(iterator)  # エピソードの取り出し
            loss_info = agent.train(experience=experience)  # 学習
            # lossと報酬の計算
            episode_average_loss.append(loss_info.loss.numpy())
            episode_rewards += R
            # 終了判定
            if next_time_step.is_last():  # 設定した行動回数に達したか？
                break
            else:
                time_step = next_time_step  # 次の状態を現在の状態にする
        # 行動終了後の情報の表示
        print(
            f'Episode:{episode + 1}, Rewards:{episode_rewards}, Average Loss:{np.mean(episode_average_loss):.6f}, Current Epsilon: {policy._epsilon:.6f}')
    # ポリシーの保存
    tf_policy_saver = policy_saver.PolicySaver(policy=agent.policy)
    tf_policy_saver.save(export_dir='policy')


if __name__ == '__main__':
    main()
