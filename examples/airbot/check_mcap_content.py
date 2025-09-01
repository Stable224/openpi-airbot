#!/usr/bin/env python3
"""
检查MCAP文件内容的脚本
"""

import sys
from pathlib import Path
from mcap.reader import make_reader

def check_mcap_file(mcap_path: str):
    """检查MCAP文件内容"""
    mcap_file = Path(mcap_path)
    if not mcap_file.exists():
        print(f"文件不存在: {mcap_path}")
        return
    
    print(f"检查文件: {mcap_path}")
    print("=" * 50)
    
    with mcap_file.open("rb") as f:
        reader = make_reader(f)
        
        # 检查附件
        print("附件列表:")
        attachment_count = 0
        for attach in reader.iter_attachments():
            attachment_count += 1
            print(f"  - 名称: {attach.name}")
            print(f"    媒体类型: {attach.media_type}")
            print(f"    大小: {len(attach.data)} bytes")
            print()
        
        if attachment_count == 0:
            print("  没有找到附件")
        print()
        
        # 检查频道/主题
        print("频道/主题列表:")
        channels = {}
        for schema, channel, message in reader.iter_messages():
            if channel.topic not in channels:
                channels[channel.topic] = {
                    'schema': schema.name,
                    'count': 0,
                    'channel_id': channel.id
                }
            channels[channel.topic]['count'] += 1
            
            # 只显示前几个消息的详细信息
            if channels[channel.topic]['count'] <= 3:
                print(f"  - 主题: {channel.topic}")
                print(f"    Schema: {schema.name}")
                print(f"    频道ID: {channel.id}")
                print(f"    消息时间戳: {message.log_time}")
                if channels[channel.topic]['count'] == 1:
                    print(f"    消息数据大小: {len(message.data)} bytes")
                print()
        
        print("主题摘要:")
        for topic, info in channels.items():
            print(f"  - {topic}: {info['count']} 条消息 (Schema: {info['schema']})")
        
        print(f"\n总计: {len(channels)} 个主题, {attachment_count} 个附件")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_mcap_content.py <mcap文件路径>")
        sys.exit(1)
    
    check_mcap_file(sys.argv[1]) 