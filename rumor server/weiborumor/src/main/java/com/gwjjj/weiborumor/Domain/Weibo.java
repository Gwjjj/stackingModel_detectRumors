package com.gwjjj.weiborumor.Domain;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Weibo {
    private String userName;

    private String createTime;

    private String text;

    private String prob;

    private String userPicUrl;

    private String comment;

    private String zan;

    private String id;
}
