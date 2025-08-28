<script lang="ts" setup>
import { ref } from 'vue'
import { UploadOutlined, MessageOutlined, MenuUnfoldOutlined, MenuFoldOutlined } from '@ant-design/icons-vue'
import UploadView from '@/components/UploadView.vue'
import ChatView from '@/components/ChatView.vue'

const collapsed = ref(false)
const selectedKeys = ref<string[]>(['1'])
</script>

<template>
  <a-layout style="min-height: 100vh">
    <a-layout-sider v-model:collapsed="collapsed" collapsible breakpoint="md" :collapsedWidth="0">
      <div class="logo">AI知识库</div>
      <a-menu v-model:selectedKeys="selectedKeys" theme="dark" mode="inline">
        <a-menu-item key="1">
          <upload-outlined />
          <span>知识文件上传</span>
        </a-menu-item>
        <a-menu-item key="2">
          <message-outlined />
          <span>大模型对话</span>
        </a-menu-item>
      </a-menu>
    </a-layout-sider>
    <a-layout>
      <a-layout-header class="header">
        <a-button class="sider-toggle" type="text" @click="collapsed = !collapsed">
          <menu-unfold-outlined v-if="collapsed" />
          <menu-fold-outlined v-else />
        </a-button>
        <span class="title">Youndon Tsing's LLM</span>
      </a-layout-header>
      <a-layout-content class="content">
        <div class="card">
          <UploadView v-show="selectedKeys[0] === '1'" />
          <ChatView v-show="selectedKeys[0] === '2'" />
  </div>
      </a-layout-content>
      <a-layout-footer class="footer">Youndon Tsing ©2025 Created with Ant Design Vue</a-layout-footer>
    </a-layout>
  </a-layout>
</template>

<style scoped>
.logo {
  height: 32px;
  background: rgba(255, 255, 255, 0.2);
  margin: 16px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: bold;
}
.header {
  background: #fff;
  padding: 0 16px;
  display: flex;
  align-items: center;
}
.sider-toggle { margin-right: 8px; }
@media (min-width: 768px) { .sider-toggle { display: none; } }
.title {
  font-size: 18px;
  font-weight: bold;
}
.content {
  margin: 16px;
}
.card {
  padding: 16px;
  background: #fff;
  border-radius: 8px;
  width: 100%;
}
.footer {
  text-align: center;
}
@media (max-width: 768px) {
  .content { margin: 8px; }
  .card { padding: 12px; min-height: 240px; }
  .title { font-size: 16px; }
}
</style>
